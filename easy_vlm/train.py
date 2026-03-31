import pathlib
import warnings
import torch
from transformers.trainer_utils import enable_full_determinism, set_seed
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    CONFIG_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    PROCESSOR_MAPPING,
)
try:
    from transformers import AutoVideoProcessor
except ImportError:
    AutoVideoProcessor = None

from easy_vlm.training import (
    get_args,
    SFTDataset,
    TrainingArguments,
    DataCollator,
    Trainer,
    rank0_print,
    check_chat_template,
    find_all_linear_names
)
import sys
sys.path.append('./')
from easy_vlm.models.qwen3vl_seg import Qwen3VLSegForConditionalGeneration
from easy_vlm.models import (
    _load_manual_qwen3vl_tokenizer_and_processor,
    load_sam3_seg_processor,
)
from easy_vlm.constants import (REGION_TOKEN, SEG_TOKEN, REF_START_TOKEN, REF_END_TOKEN, SEG_START_TOKEN, SEG_END_TOKEN)
from .nv import *
from easy_vlm.models.attention_ import *

def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def warp_onelogger_trainer(one_logger_callback_utils):
    from one_logger_utils.huggingface import hook_trainer_cls
    CustomizedTrainer = hook_trainer_cls(Trainer, one_logger_callback_utils=one_logger_callback_utils)
    return CustomizedTrainer


def build_model(args: TrainingArguments):
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

    original_config = AutoConfig.from_pretrained(args.model_path)
    if hasattr(original_config, "text_config"):
        rope_parameters = getattr(original_config.text_config, "rope_parameters", None)
        rope_scaling = getattr(original_config.text_config, "rope_scaling", None)
        if rope_scaling is None and rope_parameters is not None:
            original_config.text_config.rope_scaling = dict(rope_parameters)
    enable_full_determinism(args.seed) if args.full_determinism else set_seed(args.seed)

    processor = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    except Exception as exc:
        warnings.warn(
            f"Fast tokenizer load failed for {args.model_path}: {exc}. Falling back to use_fast=False."
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        except Exception as slow_exc:
            warnings.warn(
                f"Slow tokenizer load failed for {args.model_path}: {slow_exc}. "
                "Falling back to manual Qwen3-VL tokenizer/processor construction."
            )
            tokenizer, processor = _load_manual_qwen3vl_tokenizer_and_processor(args.model_path)

    if processor is None:
        image_processor = AutoImageProcessor.from_pretrained(args.model_path)
        if AutoVideoProcessor is not None:
            video_processor = AutoVideoProcessor.from_pretrained(
                args.model_path,
                use_token_compression=args.use_token_compression,
            )
            processor = AutoProcessor.from_pretrained(
                args.model_path,
                tokenizer=tokenizer,
                image_processor=image_processor,
                video_processor=video_processor,
            )
        else:
            processor = AutoProcessor.from_pretrained(args.model_path)
            video_processor = getattr(processor, "video_processor", None)
            if video_processor is None:
                raise ImportError(
                    "AutoVideoProcessor is unavailable in this transformers build, "
                    "and AutoProcessor did not expose a video_processor fallback."
                )
            if hasattr(video_processor, "use_token_compression"):
                video_processor.use_token_compression = args.use_token_compression
    else:
        image_processor = processor.image_processor
        video_processor = processor.video_processor
        if hasattr(video_processor, "use_token_compression"):
            video_processor.use_token_compression = args.use_token_compression

    processor.tokenizer.add_tokens([REGION_TOKEN], special_tokens=True)
    processor.tokenizer.add_tokens([SEG_TOKEN, REF_START_TOKEN, REF_END_TOKEN, SEG_START_TOKEN, SEG_END_TOKEN], special_tokens=True)

    original_config.region_token_index = processor.tokenizer.convert_tokens_to_ids(REGION_TOKEN)
    original_config.seg_token_index = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    original_config.seg_start_token_index = processor.tokenizer.convert_tokens_to_ids(SEG_START_TOKEN)
    original_config.seg_end_token_index = processor.tokenizer.convert_tokens_to_ids(SEG_END_TOKEN)
    original_config.ref_start_token_index = processor.tokenizer.convert_tokens_to_ids(REF_START_TOKEN)
    original_config.ref_end_token_index = processor.tokenizer.convert_tokens_to_ids(REF_END_TOKEN)


    original_config.max_seg_nums = args.max_seg_nums
    original_config.seg_encoder = args.seg_encoder
    original_config.seg_decoder = args.seg_decoder
    original_config.mask_decoder_model = args.mask_decoder_model
    original_config.dice_loss_weight = 0.5
    original_config.bce_loss_weight = 2.0
    original_config.cls_loss_weight = 1.0
    original_config.loss_sample_points = args.loss_sample_points

    model = Qwen3VLSegForConditionalGeneration.from_pretrained(
        args.model_path,
        config=original_config,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        # Building the SAM3 grounding branch inside HF's meta-init context can
        # crash on tensor.item()/torch.linspace paths. Use a real materialized
        # load during training model construction.
        low_cpu_mem_usage=False,
    )

    video_propagation_trainer = None
    if args.mask_decoder_model is not None:
        video_propagation_trainer = model.initialize_video_propagation_trainer()


    if args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if args.bits == 16:
            if args.bf16:
                model.to(torch.bfloat16)
            if args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if args.mask_decoder_model is not None:
        seg_processor = load_sam3_seg_processor(args.mask_decoder_model)
    else:
        seg_processor = None

    if args.mask_decoder_model is not None and 'mm_mask_decoder' not in model.get_model().config:
        print('initialize mask decoder...')
        model.get_model().initialize_mask_decoder(model.get_model().config)

    if args.mask_decoder_model is not None:
        sam_num_queries = model.get_model().grounding_model.get_num_queries()
        if model.get_model().config.max_seg_nums > sam_num_queries:
            raise ValueError(
                "max_seg_nums cannot exceed the sam3_full decoder query count. "
                f"got max_seg_nums={model.get_model().config.max_seg_nums}, "
                f"sam3_full.num_queries={sam_num_queries}."
            )

    if args.mask_decoder_model is not None and (args.bf16 or args.fp16):
        # Keep the SAM3-full grounding path in fp32 for training stability.
        # Its geometry / positional-encoding stack still relies on float kernels.
        model.get_model().grounding_model.float()
        # The query/text projection heads feed the fp32 grounding decoder and can
        # overflow in bf16 on video samples. Keep them in fp32 as well.
        model.get_model().text_hidden_fcs.float()
        model.get_model().mask_hidden_fcs.float()
        model.get_model().video_query_projector.float()

    if hasattr(model.get_model(), "video_query_alpha"):
        # Video finetuning is designed to start from the learned image-query
        # space and only learn a gated residual on top. Make that explicit at
        # training build time so stale checkpoint values cannot destabilize the
        # first forward.
        with torch.no_grad():
            model.get_model().video_query_alpha.zero_()

    if video_propagation_trainer is not None and (args.bf16 or args.fp16):
        video_propagation_trainer.float()

    # for p in model.get_model().parameters():
    #     p.requires_grad = True

    if args.llm_lr is None or args.llm_lr==0:
        for p in model.get_model().language_model.parameters():
            p.requires_grad = False

    if args.vision_encoder_lr is None or args.vision_encoder_lr==0:
        for p in model.get_model().visual.parameters():
            p.requires_grad = False

    if args.projector_lr is None or args.projector_lr==0:
        for p in model.get_model().visual.merger.parameters():
            p.requires_grad = False
    else:
        for p in model.get_model().visual.merger.parameters():
            p.requires_grad = True

    sam_model = model.get_model().grounding_model.get_sam_model()
    sam_vision_encoder = model.get_model().grounding_model.get_sam_vision_encoder()

    if args.sam_decoder_lr is None or args.sam_decoder_lr==0:
        for p in sam_model.parameters():
            p.requires_grad = False
        if video_propagation_trainer is not None:
            for p in video_propagation_trainer.sam3_video_model.detector.parameters():
                p.requires_grad = False
            for p in video_propagation_trainer.sam3_video_model.tracker.parameters():
                p.requires_grad = False
    else:
        for p in sam_model.parameters():
            p.requires_grad = True
        if video_propagation_trainer is not None:
            for p in video_propagation_trainer.sam3_video_model.detector.parameters():
                p.requires_grad = True
            for p in video_propagation_trainer.sam3_video_model.tracker.parameters():
                p.requires_grad = True

    if args.sam_encoder_lr is None or args.sam_encoder_lr==0:
        for p in sam_vision_encoder.parameters():
            p.requires_grad = False
        if video_propagation_trainer is not None:
            for p in video_propagation_trainer.sam3_video_model.detector.backbone.vision_backbone.parameters():
                p.requires_grad = False
    else:
        for p in sam_vision_encoder.parameters():
            p.requires_grad = True
        if video_propagation_trainer is not None:
            for p in video_propagation_trainer.sam3_video_model.detector.backbone.vision_backbone.parameters():
                p.requires_grad = True

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in [
                    "lm_head",
                    "embed_tokens",
                    "text_hidden_fcs",
                    "mask_hidden_fcs",
                    "video_query_projector",
                    "video_query_alpha",
                    "mask_queries",
                ]
            ]
        ):
            # print(n)
            p.requires_grad = True
        if args.mask_queries_grad is False and "mask_queries" in n:
            p.requires_grad = False
    
    # print('requires grad params:')
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name)
    # print('*****************')
    # import pdb 
    # pdb.set_trace()
    check_chat_template(processor)

    def mark_language_model_modules(model):
        for name, m in model.named_modules():
            if name.startswith("language_model."):
                setattr(m, "_is_in_language_model", True)

    mark_language_model_modules(model)

    return model, processor, seg_processor


def train():
    set_seed(42)
    args = get_args()

    if args.use_onelogger:
        one_logger_callback_utils = create_onelogger_config(args, args)

    if args.use_onelogger:
        one_logger_callback_utils.on_model_init_start()

    model, processor, seg_processor = build_model(args)

    if args.use_onelogger:
        one_logger_callback_utils.on_model_init_end()


    train_dataset = SFTDataset(
        model_config=model.config,
        processor=processor,
        seg_processor=seg_processor,
        model_max_length=args.model_max_length,
        mm_max_length=args.mm_max_length,
        fps=args.fps,
        max_frames=args.max_frames,
        dataloader_num_workers=args.dataloader_num_workers,
        data_args=args,
        requires_length=args.dynamic_batching or args.decoder_load_balancing,
        use_multi_objs=args.use_multi_objs
    )

    rank0_print(
        f"Model config: {model.config}\n\nModel: {model}\n\nProcessor: {processor}\n\n"
    )

    data_collator = DataCollator(
        processor=processor,
        sequence_packing=args.sequence_packing,
    )

    my_callbacks = []
    my_callbacks.append(MemoryLoggerCallback())

    if args.use_onelogger:
        CustomTrainer = warp_onelogger_trainer(one_logger_callback_utils)
    else:
        CustomTrainer = Trainer

    trainer = CustomTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        processing_class=processor,
        callbacks=my_callbacks,
    )


    resume_from_checkpoint = len(list(pathlib.Path(args.output_dir).glob("checkpoint-*"))) > 0
    return trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    train()
