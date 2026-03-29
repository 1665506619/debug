from collections import defaultdict
from typing import Optional, Union, List, Dict

import torch
import torch.nn as nn
import transformers
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.video_llama_3.processing_video_llama_3 import VideoLlama3ProcessorKwargs
from transformers.models.video_llama_3.video_processing_video_llama_3 import smart_resize
from transformers.models.video_llama_3.configuration_video_llama_3 import VideoLlama3Config
from transformers.models.video_llama_3.modeling_video_llama_3 import (
    VideoLlama3VisionModel as _VideoLlama3VisionModel,
    VideoLlama3Model as _VideoLlama3Model,
    VideoLlama3ForConditionalGeneration as _VideoLlama3ForConditionalGeneration,
    VideoLlama3ModelOutputWithPast,
    VideoLlama3CausalLMOutputWithPast,
    VideoLlama3Projector,
)
from transformers.models.video_llama_3.processing_video_llama_3 import VideoLlama3Processor as _VideoLlama3Processor
from transformers.utils.generic import TransformersKwargs, check_model_inputs, can_return_tuple
from transformers.cache_utils import Cache
from transformers.processing_utils import AllKwargsForChatTemplate, Unpack, BatchFeature, MultiModalData

from .utils import load_multimodal_data, cross_entropy_loss, EncoderLoadBalancingHandler


class VideoLlama3VisionModel(_VideoLlama3VisionModel):
    @check_model_inputs(tie_last_hidden_states=False)
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        merge_sizes: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutput]:
        position_embeddings = self.rotary_pos_emb(grid_thw, merge_sizes)
        position_embeddings = torch.cat(position_embeddings, dim=-1)

        handler = EncoderLoadBalancingHandler(grid_thw=grid_thw, merge_size=self.spatial_merge_size)
        pixel_values = handler.preprocess(pixel_values.type(self.dtype))
        position_embeddings = handler.preprocess(position_embeddings).chunk(2, dim=-1)

        if handler.activated:
            cu_seqlens = grid_thw.new_tensor(handler.cu_seqlens, dtype=torch.int32)
        else:
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0,
                # Select dtype based on the following factors:
                #  - FA2 requires that cu_seqlens_q must have dtype int32
                #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
                # See https://github.com/huggingface/transformers/pull/34852 for more information
                dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            )
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = self.embeddings(pixel_values)
        encoder_outputs: BaseModelOutput = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        last_hidden_state = handler.postprocess(last_hidden_state)
        last_hidden_state = self.pixel_unshuffle(last_hidden_state, grid_thw, merge_sizes)

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class VideoLlama3Model(_VideoLlama3Model):
    def __init__(self, config: VideoLlama3Config):
        super(_VideoLlama3Model, self).__init__(config)
        self.vision_model = AutoModel.from_config(config.vision_config)
        self.projector = VideoLlama3Projector(config)
        self.language_model = AutoModel.from_config(config.text_config)

        in_channels = self.config.vision_config.num_channels
        patch_size = self.config.vision_config.patch_size
        self.visual_in_channels = patch_size * patch_size * in_channels

        self.post_init()

    def get_multimodal_features(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_merge_sizes: Optional[torch.LongTensor] = None,
    ):
        if pixel_values is None:
            pixel_values = torch.zeros(
                0, self.visual_in_channels, dtype=self.vision_model.dtype, device=self.vision_model.device
            )
            image_grid_thw = torch.zeros((0, 3), dtype=torch.long, device=self.vision_model.device)
            image_merge_sizes = torch.zeros((0,), dtype=torch.long, device=self.vision_model.device)

        if pixel_values_videos is None:
            pixel_values_videos = torch.zeros(
                0, self.visual_in_channels, dtype=self.vision_model.dtype, device=self.vision_model.device
            )
            video_grid_thw = torch.zeros((0, 3), dtype=torch.long, device=self.vision_model.device)
            video_merge_sizes = torch.zeros((0,), dtype=torch.long, device=self.vision_model.device)

        pixel_values = torch.cat([pixel_values, pixel_values_videos], dim=0).type(self.vision_model.dtype)
        grid_thw = torch.cat([image_grid_thw, video_grid_thw], dim=0)
        merge_sizes = torch.cat([image_merge_sizes, video_merge_sizes], dim=0)
        visual_embeds = self.vision_model(pixel_values, grid_thw=grid_thw, merge_sizes=merge_sizes).last_hidden_state
        visual_embeds = self.projector(visual_embeds)

        num_image_tokens = torch.sum(image_grid_thw.prod(dim=1) // image_merge_sizes)
        image_embeds = visual_embeds[:num_image_tokens]
        video_embeds = visual_embeds[num_image_tokens:]

        return image_embeds, video_embeds

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_merge_sizes: Optional[torch.LongTensor] = None,
        video_compression_mask: Optional[torch.BoolTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, VideoLlama3ModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = video_embeds = None
        if self.training or pixel_values is not None or pixel_values_videos is not None:
            image_embeds, video_embeds = self.get_multimodal_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_merge_sizes=image_merge_sizes,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                video_merge_sizes=video_merge_sizes,
            )

            image_mask, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            if video_compression_mask is not None:
                video_embeds = video_embeds[video_compression_mask.to(video_embeds.device)]
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return VideoLlama3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_embeds,
            video_hidden_states=video_embeds,
        )


class VideoLlama3ForConditionalGeneration(_VideoLlama3ForConditionalGeneration):
    accepts_loss_kwargs = True

    def __init__(self, config: VideoLlama3Config):
        super(_VideoLlama3ForConditionalGeneration, self).__init__(config)
        self.model = VideoLlama3Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_merge_sizes: Optional[torch.LongTensor] = None,
        video_compression_mask: Optional[torch.BoolTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, VideoLlama3CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_merge_sizes=image_merge_sizes,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            video_merge_sizes=video_merge_sizes,
            video_compression_mask=video_compression_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        loss, logits = None, None
        if labels is not None:
            loss = cross_entropy_loss(
                hidden_states=hidden_states,
                lm_head=self.lm_head,
                position_ids=position_ids,
                labels=labels,
                **kwargs,
            )
            if pixel_values is None and pixel_values_videos is None:
                num_channels = self.config.vision_config.num_channels
                patch_size = self.config.vision_config.patch_size
                pixel_values = torch.zeros(
                    (1, 1, patch_size * patch_size * num_channels),
                    dtype=self.dtype,
                    device=self.device,
                )
                image_grid_thw = torch.ones((1, 3), dtype=torch.long, device=self.device)
                image_merge_sizes = torch.ones((1,), dtype=torch.long, device=self.device)
                image_embeds = self.model.get_image_features(pixel_values, image_grid_thw, image_merge_sizes)
                loss = loss + torch.cat(image_embeds, dim=0).sum() * 0.0
        else:
            logits = self.lm_head(hidden_states)

        return VideoLlama3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )


class VideoLlama3Processor(_VideoLlama3Processor):
    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        chat_template: Optional[str] = None,
        mm_max_length: Optional[int] = None,
        return_labels: bool = False,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ):
        if return_labels:
            assert kwargs.get("return_tensors", None) == "pt", (
                "`return_tensors` must be set to `pt` when `return_labels` is True."
            )
            assert not kwargs.get("add_generation_prompt", False), (
                "`add_generation_prompt` must be set to False when `return_labels` is True."
            )
            assert kwargs.get("tokenize", True), "`tokenize` must be set to True when `return_labels` is True."
            assert kwargs.get("return_dict", False), "`return_dict` must be set to True when `return_labels` is True."

            pseudo_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
            prompt_tokens = super().apply_chat_template(
                pseudo_message, chat_template=chat_template, tokenize=True, add_generation_prompt=False
            )[0]
            conv_tokens = super().apply_chat_template(
                pseudo_message, chat_template=chat_template, tokenize=True, add_generation_prompt=True
            )[0]
            prompt_length = len(conv_tokens) - len(prompt_tokens)

            ignore_tokens = torch.as_tensor([self.image_token_id, self.video_token_id])[None, None]

        fps = kwargs.pop("fps", 1)
        max_frames = kwargs.pop("max_frames", None)
        tokenize = kwargs.pop("tokenize", True)
        return_dict = kwargs.pop("return_dict", False)
        return_tensors = kwargs.pop("return_tensors", None)
        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        kwargs.pop("do_sample_frames", False)

        if tokenize and return_dict:
            conversation = load_multimodal_data(
                conversation,
                fps=fps,
                max_frames=max_frames,
            )

            if mm_max_length is not None:
                assert "max_pixels" not in kwargs and "size" not in kwargs, (
                    "Please provide only one of `mm_max_length` and `max_pixels`."
                )
                num_images, num_videos = 0, 0
                for message in conversation:
                    for content in message["content"]:
                        if content["type"] == "image":
                            num_images += 1
                        elif content["type"] == "video":
                            num_videos += 1
                kwargs["max_pixels"] = self._get_max_pixels(
                    num_images=num_images,
                    num_videos=num_videos,
                    mm_max_length=mm_max_length,
                )

        outputs = defaultdict(list)

        for i, message in enumerate(conversation):
            prompt = super().apply_chat_template(
                [message],
                chat_template=chat_template,
                tokenize=False,
                add_generation_prompt=add_generation_prompt and i == len(conversation) - 1,
            )

            if tokenize and return_dict:
                images, videos, video_metadatas = [], [], []
                if message["role"] != "assistant":
                    for content in message["content"]:
                        if content["type"] == "image":
                            images.append(content["image"])
                        elif content["type"] == "video":
                            videos.append(content["video"][0])
                            video_metadatas.append(content["video"][1])

                results = self(
                    text=prompt,
                    images=images if len(images) > 0 else None,
                    videos=videos if len(videos) > 0 else None,
                    video_metadata=video_metadatas if len(videos) > 0 else None,
                    return_tensors="pt",
                    do_sample_frames=False,
                    **kwargs,
                )

                if return_labels:
                    labels = torch.full_like(results["input_ids"], fill_value=-100, dtype=torch.long)
                    if message["role"] == "assistant":
                        valid_mask = torch.all(results["input_ids"][..., None] != ignore_tokens, dim=-1)
                        # prefix: <|im_start|>assistant\n
                        valid_mask[:, :prompt_length] = False
                        # postfix: \n
                        valid_mask[:, -1] = False
                        labels[valid_mask] = results["input_ids"][valid_mask]
                    results["labels"] = labels

                for key, value in results.items():
                    outputs[key].append(value)

            else:
                outputs["prompts"].append(prompt)

        if tokenize:
            mm_input_names = set(self.image_processor.model_input_names + self.video_processor.model_input_names)
            for k, v in outputs.items():
                if k in mm_input_names:
                    outputs[k] = torch.cat(v, dim=0)
                else:
                    outputs[k] = torch.cat(v, dim=1)
            outputs = BatchFeature(outputs, tensor_type=return_tensors)
            if return_dict:
                return outputs
            return outputs["input_ids"]

        return "".join(outputs["prompts"])

    def _get_max_pixels(
        self,
        num_images: int,
        num_videos: int,
        mm_max_length: Optional[int] = None,
    ):
        merge_size = max(self.image_processor.merge_size, self.video_processor.merge_size)
        if num_images > 0:
            merge_size = min(merge_size, self.image_processor.merge_size)
        if num_videos > 0:
            merge_size = min(merge_size, self.video_processor.merge_size)
        factor = self.image_processor.patch_size * merge_size
        return mm_max_length // max(num_images + num_videos, 1) * (factor**2)

    def _get_number_of_video_patches(self, num_frames: int, height: int, width: int, videos_kwargs=None):
        min_pixels = videos_kwargs.get("min_pixels", None) or self.video_processor.size["shortest_edge"]
        max_pixels = videos_kwargs.get("max_pixels", None) or self.video_processor.size["longest_edge"]
        patch_size = videos_kwargs.get("patch_size", None) or self.video_processor.patch_size
        merge_size = videos_kwargs.get("merge_size", None) or self.video_processor.merge_size

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels // num_frames
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return num_frames * grid_h * grid_w

    def _get_num_multimodal_tokens(
        self,
        image_sizes=None,
        video_sizes=None,
        mm_max_length: Optional[int] = None,
        **kwargs,
    ):
        if mm_max_length is not None:
            assert "max_pixels" not in kwargs, "Please provide only one of `mm_max_length` and `max_pixels`."
            kwargs["max_pixels"] = self._get_max_pixels(
                num_images=len(image_sizes) if image_sizes is not None else 0,
                num_videos=len(video_sizes) if video_sizes is not None else 0,
                mm_max_length=mm_max_length,
            )

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = VideoLlama3ProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = VideoLlama3ProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            merge_size = videos_kwargs.get("merge_size", None) or self.video_processor.merge_size

            fps = kwargs.pop("fps", 1)
            max_frames = kwargs.pop("max_frames", None)
            for video_size in video_sizes:
                num_frames = video_size[0] // fps
                if max_frames is not None:
                    num_frames = min(num_frames, max_frames)
                video_size[0] = num_frames

            num_video_patches = [
                self._get_number_of_video_patches(*video_size, videos_kwargs) for video_size in video_sizes
            ]
            num_video_tokens = [(num_patches // merge_size**2) for num_patches in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)


transformers.models.video_llama_3.modeling_video_llama_3.VideoLlama3ForConditionalGeneration = (
    VideoLlama3ForConditionalGeneration
)
transformers.models.auto.modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[VideoLlama3Config] = (
    VideoLlama3ForConditionalGeneration
)

transformers.models.video_llama_3.processing_video_llama_3.VideoLlama3Processor = VideoLlama3Processor
transformers.models.auto.processing_auto.PROCESSOR_MAPPING[VideoLlama3Config] = VideoLlama3Processor
