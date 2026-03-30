from . import qwen3_vl

try:
    from . import video_llama_3
except Exception:
    import warnings

    warnings.warn("Fail to import `VideoLlama3` implementation from `transformers`.")
import torch
import os
import json
import warnings
import numpy as np
from .qwen3vl_seg import Qwen3VLSegForConditionalGeneration
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoImageProcessor,
    AutoVideoProcessor,
    AutoTokenizer,
    CONFIG_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    PROCESSOR_MAPPING,
)
from peft import PeftConfig
from safetensors.torch import load_file
from .attention_ import *
from .video_seg_engine import VideoSegEngine


class _ManualSam3SegProcessor:
    """Lightweight SAM3 image preprocessor aligned with sam3_full transforms."""

    def __init__(self, resolution=1008):
        from PIL import Image
        from torchvision.transforms import v2

        self._pil_image_cls = Image.Image
        self.resolution = resolution
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    class _Output(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    def __call__(self, *args, **kwargs):
        from torchvision.transforms import v2

        if len(args) == 0:
            raise ValueError("SAM3 processor expects an image input")
        image = args[0]

        if isinstance(image, self._pil_image_cls):
            width, height = image.size
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        elif torch.is_tensor(image):
            height, width = image.shape[-2:]
        else:
            raise ValueError(f"Unsupported image type for SAM3 preprocessing: {type(image)}")

        tensor = v2.functional.to_image(image)
        pixel_values = self.transform(tensor).unsqueeze(0)
        return self._Output(
            {
                "pixel_values": pixel_values,
                "original_sizes": [(height, width)],
            }
        )

    def __getattr__(self, name):
        raise AttributeError(name)

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-') or 'bak' in model_paths[-1]:
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def _load_manual_qwen3vl_tokenizer_and_processor(model_path):
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    tokenizer_kwargs = {
        "tokenizer_file": os.path.join(model_path, "tokenizer.json"),
        "clean_up_tokenization_spaces": False,
    }
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        if tokenizer_config.get("eos_token") is not None:
            tokenizer_kwargs["eos_token"] = tokenizer_config["eos_token"]
        if tokenizer_config.get("pad_token") is not None:
            tokenizer_kwargs["pad_token"] = tokenizer_config["pad_token"]
        extra_special_tokens = tokenizer_config.get("extra_special_tokens")
        if isinstance(extra_special_tokens, list):
            tokenizer_kwargs["additional_special_tokens"] = extra_special_tokens

    tokenizer = Qwen2TokenizerFast(**tokenizer_kwargs)
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    video_processor = AutoVideoProcessor.from_pretrained(model_path)

    chat_template = None
    for candidate in ["chat_template.jinja", "chat_template.json"]:
        path = os.path.join(model_path, candidate)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                chat_template = f.read()
            break

    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=chat_template,
    )
    return tokenizer, processor


def load_sam3_seg_processor(mask_decoder_model):
    local_files_only = bool(mask_decoder_model and os.path.exists(mask_decoder_model))
    try:
        return AutoProcessor.from_pretrained(mask_decoder_model, local_files_only=local_files_only)
    except Exception as exc:
        warnings.warn(
            f"SAM3 processor load failed for {mask_decoder_model}: {exc}. "
            "Falling back to a local sam3_full-style image preprocessor."
        )
        return _ManualSam3SegProcessor()

def load_pretrained_model(model_path, model_base, load_8bit=False, load_4bit=False, device_map="auto", **kwargs):
    model_name = get_model_name_from_path(model_path)
    if 'token' in kwargs:
        token = kwargs['token']
    else:
        token = None
    
    save_path = kwargs.pop('save_path', False)

    # NOTE: auto device_map by default
    # if want to put model into a single device, you can set device_map={"": "cuda:0"}
    kwargs = {"device_map": device_map, **kwargs}

    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, "text_config"):
        rope_parameters = getattr(config.text_config, "rope_parameters", None)
        rope_scaling = getattr(config.text_config, "rope_scaling", None)
        if rope_scaling is None and rope_parameters is not None:
            config.text_config.rope_scaling = dict(rope_parameters)
    config._attn_implementation = kwargs.pop('attn_implementation', "flash_attention_2") # default to flash_attention_2

    torch_dtype = config.torch_dtype if hasattr(config, "torch_dtype") else kwargs.pop('torch_dtype', torch.float16)

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # NOTE: High-version Transformers will report: """ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time."""
        # kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch_dtype
    # judge model type
    model_type = config.model_type if hasattr(config, "model_type") else kwargs.pop('model_type', "videollama3_qwen2")

    # judge pretrain/finetune
    is_alignment = getattr(config, "tune_mm_mlp_adapter", False) or getattr(config, "is_alignment", False)

    # NOTE: lora/qlora model loading
    processor = None

    is_peft_checkpoint = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_peft_checkpoint or 'lora' in model_name.lower() or 'qlora' in model_name.lower():
    # if True:
        cfg_pretrained = PeftConfig.from_pretrained(model_path, token=token)
        # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
        # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
        model_base = model_base if model_base is not None else cfg_pretrained.base_model_name_or_path

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)
        except Exception as exc:
            warnings.warn(
                f"Fast tokenizer load failed for {model_path}: {exc}. Falling back to use_fast=False."
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
            except Exception as slow_exc:
                warnings.warn(
                    f"Slow tokenizer load failed for {model_path}: {slow_exc}. "
                    "Falling back to manual Qwen3-VL tokenizer/processor construction."
                )
                tokenizer, processor = _load_manual_qwen3vl_tokenizer_and_processor(model_path)
        print('Loading Qwen from base model...')
        print(model_base)
        
        model = Qwen3VLSegForConditionalGeneration.from_pretrained(
            model_base, 
            low_cpu_mem_usage=True, 
            config=config, 
            ignore_mismatched_sizes=True, 
            attn_implementation="fused_attention",
            **kwargs
        )

        print('Loading additional Qwen3 weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu', weights_only=True,)
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')
            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    
        frozen_path = os.path.join(config.mask_decoder_model, 'model.safetensors')
        if os.path.exists(frozen_path):
            non_lora_frozen = load_file(frozen_path)
            for k, v in non_lora_frozen.items():
                k = k.replace('detector_model', 'model.grounding_model.model')
                if k not in non_lora_trainables:
                    non_lora_trainables[k] = v
        else:
            warnings.warn(
                f"SAM3 frozen weights not found at {frozen_path}; "
                "continuing without loading grounding-model frozen parameters."
            )
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')


        def mark_language_model_modules(model):
            for name, m in model.named_modules():
                if name.startswith("language_model."):
                    setattr(m, "_is_in_language_model", True)

        mark_language_model_modules(model)



    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)
        except Exception as exc:
            warnings.warn(
                f"Fast tokenizer load failed for {model_path}: {exc}. Falling back to use_fast=False."
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
            except Exception as slow_exc:
                warnings.warn(
                    f"Slow tokenizer load failed for {model_path}: {slow_exc}. "
                    "Falling back to manual Qwen3-VL tokenizer/processor construction."
                )
                tokenizer, processor = _load_manual_qwen3vl_tokenizer_and_processor(model_path)
        model = Qwen3VLSegForConditionalGeneration.from_pretrained(model_path, config=config, **kwargs)

    if processor is None:
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
            )
        except Exception as exc:
            warnings.warn(
                f"AutoProcessor load failed for {model_path}: {exc}. "
                "Falling back to manual Qwen3-VL tokenizer/processor construction."
            )
            tokenizer, processor = _load_manual_qwen3vl_tokenizer_and_processor(model_path)
    

    if save_path:
        model.save_pretrained(save_path, state_dict=model.state_dict())
        tokenizer.save_pretrained(save_path)
        processor.save_pretrained(save_path)

    return tokenizer, model, processor
