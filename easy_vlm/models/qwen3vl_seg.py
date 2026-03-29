import math
import inspect
import os
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor as _Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel as _Qwen3VLModel,
    Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration,
    Qwen3VLVisionModel as _Qwen3VLVisionModel,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextModel,
)
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids
from transformers.processing_utils import AllKwargsForChatTemplate, Unpack, BatchFeature, MultiModalData
from transformers.utils import is_torchdynamo_compiling, can_return_tuple
from transformers.utils.generic import TransformersKwargs, check_model_inputs
from transformers.modeling_outputs import ModelOutput

from .utils import load_multimodal_data, cross_entropy_loss, EncoderLoadBalancingHandler, CrossEntropyLoss, DiceLoss, genetate_video_pred_embeddings, process_video_gt_masks, binary_focal_loss_with_logits, projection_loss, get_phrase_embedding, expand_vision_features, get_phrase_ids_by_start_end, dice_score, calculate_mask_loss_group, downsample_to_max_hw, select_vision_outputs, resize_pred_and_gt_for_loss
from .segmentation_decoder import SegmentationDecoder
from .assigner import HungarianAssigner
from easy_vlm.constants import IGNORE_INDEX
from .omni_attention import omni_attn_mask_naive, full_attn_mask, fused_full_attn_mask
from .point_sample import sample_points

@dataclass
class Qwen3VLSegModelOutputWithPast(ModelOutput):

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    labels: Optional[torch.LongTensor] = None

@dataclass
class Qwen3VLSegCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    mask_bce_loss: Optional[torch.FloatTensor] = None
    mask_dice_loss: Optional[torch.FloatTensor] = None
    mask_loss: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen3VLSegModel(_Qwen3VLModel): 
    def __init__(self, config: Qwen3VLConfig):
        super(Qwen3VLSegModel, self).__init__(config)
        if 'out_dim' not in config:
            config.out_dim = 256    
        self.build_mask_decoder(config)
        self.grounding_model = SegmentationDecoder(config)   

    @staticmethod
    def _unpack_visual_outputs(outputs):
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output, getattr(outputs, "deepstack_features", None)
        return outputs

    def initialize_mask_decoder(self, config):
        self.grounding_model.load_model(config)
        self.config.mm_mask_decoder = config.mask_decoder_model
        with torch.no_grad():
            self.mask_queries.zero_()

    def build_mask_decoder(self, config):
        # if config.training:
        # self.class_head = SegPresenceClassifier()
            
        # Projection layer for lisa
        in_dim = config.text_config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True    

        mask_fcs = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.mask_hidden_fcs = nn.ModuleList([nn.Sequential(*mask_fcs)])
        self.mask_hidden_fcs.train()
        for param in self.mask_hidden_fcs.parameters():
            param.requires_grad = True    

        video_query_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.video_query_projector = nn.Sequential(*video_query_fc)
        self.video_query_projector.train()
        for param in self.video_query_projector.parameters():
            param.requires_grad = True

        self.mask_queries = nn.Parameter(torch.zeros(config.max_seg_nums, config.text_config.hidden_size))  

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        masks: Optional[List[torch.LongTensor]] = None,
        mask_ids = None,
        sam_images = None,
        masks_valid = None,
        mask_type = None,
        labels = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self._unpack_visual_outputs(
                self.get_image_features(pixel_values, image_grid_thw)
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self._unpack_visual_outputs(
                self.get_video_features(pixel_values_videos, video_grid_thw)
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # replace [SEG] token with queries
        if input_ids is not None:
            B, N = input_ids.shape
            mask_selected = (input_ids == self.config.seg_token_index)
            modality_batch = []
            # print(mask_selected.sum())
            
            if mask_selected.sum() > 0: 
                mask_num = mask_selected.sum()//self.config.max_seg_nums
                mask_feats = self.mask_queries.repeat(mask_num,1)
                inputs_embeds[mask_selected] = inputs_embeds[mask_selected]*0.0 + mask_feats

                mask_indices = mask_selected.nonzero(as_tuple=False)  # [n, 2] -> (b, pos)
                mask_indices_right = mask_indices.clone()
                mask_indices_right[:, 1] = mask_indices_right[:, 1] + 1
                valid = mask_indices_right[:, 1] < N
                mask_indices_right = mask_indices_right[valid]

                mask_selected_right = torch.zeros_like(mask_selected)
                mask_selected_right[mask_indices_right[:, 0], mask_indices_right[:, 1]] = True
                labels[mask_selected_right] = IGNORE_INDEX # 第一个[seg]算loss 最后的<mask_end>不算loss
                labels[mask_selected] = IGNORE_INDEX

                # get start and end idx for each [SEG]
                for b in range(B):
                    row = mask_selected[b] 

                    padded = F.pad(row, (1, 1), value=False)  # [N+2]
                    diff = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)

                    starts = torch.nonzero(diff == 1, as_tuple=False).squeeze(1)  # in [0, N]
                    ends   = torch.nonzero(diff == -1, as_tuple=False).squeeze(1) # in [0, N]

                    if starts.numel() == 0:
                        spans = torch.empty((0, 2), device=input_ids.device, dtype=torch.long)
                    else:
                        spans = torch.stack([starts, ends], dim=1)  # [num_spans, 2]

                    modality_batch.append(spans)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if (
            attention_mask is not None
            and attention_mask.dim() == 2
            and len(modality_batch) > 0
        ):
            attention_mask = omni_attn_mask_naive(attention_mask, modality_batch)
        

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLSegModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
            labels=labels,
        )

class Qwen3VLSegForConditionalGeneration(_Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super(_Qwen3VLForConditionalGeneration, self).__init__(config)
        self.model = Qwen3VLSegModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
        self.video_seg_engine = None

        self.loss_mask = CrossEntropyLoss(
            use_sigmoid=True,
            reduction='mean',
            loss_weight=2.0
        )
        self.loss_dice = DiceLoss(
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=0.5
        )

        self.assigner = HungarianAssigner(
            dice_loss_weight=config.dice_loss_weight,
            ce_loss_weight=config.bce_loss_weight,
            cls_loss_weight=config.cls_loss_weight,
        )

    def get_model(self):
        return self.model

    def set_video_seg_engine(self, video_seg_engine):
        self.video_seg_engine = video_seg_engine

    def _build_video_external_query_embed(self):
        if len(self.seg_output_embeddings) == 0:
            return None
        seg_output_embeddings = torch.cat(self.seg_output_embeddings, dim=0)
        return self.model.mask_hidden_fcs[0](seg_output_embeddings)

    def _extract_ref_phrase_token_ids(self, output_ids):
        start_token_id = self.config.ref_start_token_index
        end_token_id = self.config.ref_end_token_index
        phrase_token_ids = []

        for batch_idx in range(output_ids.shape[0]):
            ids = output_ids[batch_idx]
            token_idx = 0
            while token_idx < ids.shape[0]:
                if ids[token_idx].item() != start_token_id:
                    token_idx += 1
                    continue

                end_idx = token_idx + 1
                while end_idx < ids.shape[0] and ids[end_idx].item() != end_token_id:
                    end_idx += 1
                if end_idx < ids.shape[0]:
                    phrase_token_ids.append(ids[token_idx + 1:end_idx].detach().cpu())
                token_idx = end_idx + 1

        return phrase_token_ids

    def _resolve_video_phrases(self, output_ids, tokenizer=None, phrases=None):
        phrase_texts = []
        if tokenizer is not None:
            phrase_token_ids = self._extract_ref_phrase_token_ids(output_ids)
            for token_ids in phrase_token_ids:
                phrase_text = tokenizer.decode(
                    token_ids.tolist(), skip_special_tokens=True
                ).strip()
                if phrase_text:
                    phrase_texts.append(phrase_text)

        if len(phrase_texts) > 0:
            return phrase_texts

        if phrases is None:
            raise ValueError(
                "tokenizer or explicit phrases must be provided for video inference"
            )

        if isinstance(phrases, str):
            return [phrases]
        return list(phrases)
    
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        masks: Optional[List[torch.LongTensor]] = None,
        mask_ids = None,
        sam_images = None,
        masks_valid = None,
        mask_type = None,
        phrase_ids = None,
        data_indices = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        # print('input_ids', input_ids.shape)
        # print('pixel_values', pixel_values.shape)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            masks=masks,
            mask_ids=mask_ids,
            sam_images=sam_images,
            masks_valid=masks_valid,
            mask_type=mask_type,
            labels=labels,
            **kwargs,
        )

        hidden_states = outputs['last_hidden_state']

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        ce_loss = None
        mask_bce_loss = None
        mask_dice_loss = None
        mask_loss = None
        cls_loss = None

        if labels is not None: # training
            ce_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
            
            mask_valid_ = False
            if masks[0] is not None:
                hidden_states_sam = self.model.mask_hidden_fcs[0](hidden_states)
                g_pixel_values = torch.stack(sam_images, dim=0)  # [bs, C, H, W]
                with torch.no_grad():
                    vision_outputs_batch = self.model.grounding_model.encoder(g_pixel_values)

                bs = input_ids.shape[0]
                mask_bce_sum = None
                mask_dice_sum = None
                cls_sum = None
                num_masks = 0
                num_cls = 0
                for i in range(bs):
                    pred_masks = []
                    input_id = input_ids[i]
                    seg_token_mask = input_id==self.config.seg_token_index
                    
                    pred_embedding = hidden_states_sam[i][seg_token_mask]
                    pred_embedding = pred_embedding.reshape(-1, self.config.max_seg_nums, pred_embedding.shape[-1])
                    # print('pred_embedding shape:', pred_embedding.shape) # [num_seg, max_seg_nums, dim]

                    phrase_id = input_ids.new_tensor(phrase_ids[i])
                    phrase_embedding = self.model.get_input_embeddings()(phrase_id)
                    phrase_embedding = self.model.text_hidden_fcs[0](phrase_embedding.unsqueeze(0)).squeeze(0)
                
                    gt_mask = masks[i]
                    mask_valid_ = masks_valid[i]

                    g_pixel_values = sam_images[i].unsqueeze(0)

                    vision_outputs = select_vision_outputs(vision_outputs_batch, i)

                    obj_num = pred_embedding.shape[0]

                    max_chunk = 5
                    # print('obj_num:', obj_num, 'vision_outputs shape:', vision_outputs['last_hidden_state'].shape)

                    all_mask_outputs = []
                    pred_masks_list = []
                    pred_logits_list = []

                    phrase_embedding, text_attn_mask = get_phrase_embedding(
                        phrase_id, phrase_embedding, self.config.ref_start_token_index
                    )

                    if phrase_embedding.shape[0] == 0:
                        mask_valid_ = False
                        print(data_indices)
                        print('phrase_embedding is empty')
                        break
                    else:
                        for start in range(0, obj_num, max_chunk):
                            end = min(start + max_chunk, obj_num)
                            chunk_size = end - start

                            pred_embedding_chunk = pred_embedding[start:end]  # [chunk, max_seg_nums, dim]（按你原实现）
                            vision_outputs_expand = expand_vision_features(vision_outputs, chunk_size)

                            mask_outputs_chunk = self.model.grounding_model.decoder(
                                vision_outputs_expand,
                                phrase_embedding[start:end],
                                text_attn_mask[start:end],
                                pred_embedding_chunk
                            )

                            pred_masks_chunk = mask_outputs_chunk["pred_masks"]   # [chunk, 50, H, W]
                            pred_logits_chunk = mask_outputs_chunk["pred_logits"] # [chunk, 50]

                            # resize：只处理当前 chunk（gt_mask 返回的也可直接复用）
                            pred_masks_chunk, gt_mask_rs = resize_pred_and_gt_for_loss(pred_masks_chunk, gt_mask)

                            # 对 chunk 内每个对象做 assign + loss
                            for local_midx in range(chunk_size):
                                global_midx = start + local_midx  # 对齐你原来的 midx

                                mask_id = mask_ids[i][global_midx]
                                mask_id_tensor = torch.as_tensor(mask_id, device=pred_masks_chunk.device, dtype=torch.long)

                                # 注意：尽量别 .float()，除非你的 loss 必须 FP32
                                pred_masks_cur = pred_masks_chunk[local_midx].unsqueeze(1)  # [50, 1, H, W]
                                pred_scores_cur = pred_logits_chunk[local_midx]             # [50]
                                gt_masks_cur = gt_mask_rs[mask_id_tensor]                   # [Ng, 1, H, W]

                                if gt_mask_rs.sum()>0: # not null

                                    assign_id = self.assigner.assign(
                                        pred_masks_cur.float(),
                                        gt_masks_cur.float(),
                                        pred_scores_cur.float(),
                                    )

                                    score_targets = torch.zeros_like(pred_scores_cur)
                                    for id_, asid in enumerate(assign_id):
                                        if asid != -1:
                                            gt_masks_ = gt_masks_cur[asid:asid+1]      # [1,1,H,W]
                                            pred_masks_ = pred_masks_cur[id_:id_+1]     # [1,1,H,W]

                                            if mask_type[i] == 0:  # mask
                                                if self.config.loss_sample_points:
                                                    sampled_pred_mask, sampled_gt_mask = sample_points(pred_masks_, gt_masks_)
                                                    bce = self.loss_mask(sampled_pred_mask, sampled_gt_mask)
                                                    dice = self.loss_dice(sampled_pred_mask, sampled_gt_mask)
                                                else:
                                                    bce = self.loss_mask(pred_masks_, gt_masks_)
                                                    dice = self.loss_dice(pred_masks_, gt_masks_)
                                            elif mask_type[i] == 1:  # bbox
                                                dice = projection_loss(pred_masks_, gt_masks_)
                                                bce = dice * 0.0
                                            else:
                                                raise NotImplementedError

                                            # bce = bce * _scale
                                            # dice = dice * _scale
                                            if mask_bce_sum is None:
                                                mask_bce_sum = bce
                                                mask_dice_sum = dice
                                            else:
                                                mask_bce_sum = mask_bce_sum + bce
                                                mask_dice_sum = mask_dice_sum + dice
                                            num_masks += 1

                                            q_score = dice_score(pred_masks_cur[id_].sigmoid(), gt_masks_cur[asid])
                                            score_targets[id_] = max(q_score.item(), 0.1)
                                        else:
                                            score_targets[id_] = 0.0
                                else: # null gt
                                    score_targets = torch.zeros_like(pred_scores_cur)

                                cls_ = F.binary_cross_entropy_with_logits(pred_scores_cur, score_targets, reduction="mean")
                                if cls_sum is None:
                                    cls_sum = cls_
                                else:
                                    cls_sum = cls_sum + cls_
                                num_cls += 1

                            # del mask_outputs_chunk, pred_masks_chunk, pred_logits_chunk, vision_outputs_expand, pred_embedding_chunk

                if mask_bce_sum is not None:
                    mask_bce_loss = mask_bce_sum / num_masks
                    mask_dice_loss = mask_dice_sum / num_masks
                else:
                    mask_bce_loss = ce_loss*0.0
                    mask_dice_loss = ce_loss*0.0

                if cls_sum is not None:
                    cls_loss = cls_sum / num_cls
                else:
                    cls_loss = ce_loss*0.0
                
                mask_bce_loss = self.config.bce_loss_weight * mask_bce_loss 
                mask_dice_loss = self.config.dice_loss_weight * mask_dice_loss 
                cls_loss = self.config.cls_loss_weight * cls_loss
                mask_loss = mask_bce_loss + mask_dice_loss
                loss = mask_loss + ce_loss + cls_loss
    
            if not mask_valid_:
                # print('No valid masks found.')
                loss = ce_loss
                mask_bce_loss = loss * 0.0
                mask_dice_loss = loss * 0.0
                mask_loss = loss * 0.0
                cls_loss = loss * 0.0


        if mask_loss is not None: # training
            return Qwen3VLSegCausalLMOutputWithPast(
                loss=loss,
                ce_loss=ce_loss.detach(),
                mask_bce_loss=mask_bce_loss.detach(),
                mask_dice_loss=mask_dice_loss.detach(),
                mask_loss=mask_loss.detach(),
                cls_loss=cls_loss.detach(),
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
            )
        else:
            return Qwen3VLSegCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
            )

    def inference(
        self,
        masks: Optional[List[torch.LongTensor]] = None,
        mask_ids = None,
        sam_images = None,
        masks_valid = None,
        mask_type = None,
        phrase_ids = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        self.SEG_START = None
        self.seg_output_embeddings = []
        kwargs.pop("mm_token_type_ids", None)
        kwargs.setdefault("use_cache", False)
        outputs = self.generate(
            **kwargs
        )

        input_ids = kwargs['input_ids']
        output_ids = outputs.sequences
        # last_hidden_state = []
        # for hs in outputs.hidden_states: # round
        #     last_hidden_state.append(hs[-1])
        # last_hidden_state = torch.cat(last_hidden_state, dim=1)


        pred_masks = None
        pred_logits = None
        try:
            if len(self.seg_output_embeddings)>0:
                seg_output_embeddings = torch.cat(self.seg_output_embeddings, dim=0)
                pred_embeddings = self.model.mask_hidden_fcs[0](seg_output_embeddings)

                g_pixel_values = sam_images[0].unsqueeze(0)

                vision_outputs = self.model.grounding_model.encoder(g_pixel_values)

                obj_num = pred_embeddings.shape[0]
                
                vision_outputs_expand = expand_vision_features(vision_outputs, obj_num)

                phrase_id, text_attn_mask = get_phrase_ids_by_start_end(output_ids, self.config.ref_start_token_index, self.config.ref_end_token_index)

                phrase_embedding = self.model.get_input_embeddings()(phrase_id)
                phrase_embedding = self.model.text_hidden_fcs[0](phrase_embedding)

                mask_outputs = self.model.grounding_model.decoder(vision_outputs_expand, phrase_embedding, text_attn_mask, pred_embeddings)

                pred_masks = mask_outputs['pred_masks'] # [9,10,288,288]
                pred_logits = mask_outputs['pred_logits'].sigmoid()

        except Exception as exp:
            print('Segmentation inference error:', exp)
            print(seg_output_embeddings.shape)
            print(output_ids)
            pred_masks = None
            pred_logits = None
            
        
        output_ids = output_ids[:, input_ids.shape[1]:]
        return output_ids, pred_masks, pred_logits

    def inference_video(
        self,
        video_resource_path,
        tokenizer=None,
        phrases=None,
        start_frame=0,
        video_seg_engine=None,
        video_init_kwargs=None,
        max_frame_num_to_track=None,
        propagate_both_directions=False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        self.SEG_START = None
        self.seg_output_embeddings = []
        outputs = self.generate(**kwargs)
        input_ids = kwargs["input_ids"]
        output_ids = outputs.sequences
        generated_output_ids = output_ids[:, input_ids.shape[1]:]
        video_results = None

        try:
            external_query_embeds = self._build_video_external_query_embed()
            video_seg_engine = video_seg_engine or self.video_seg_engine
            if video_seg_engine is None:
                raise ValueError(
                    "video_seg_engine must be provided before calling inference_video"
                )

            phrase_texts = self._resolve_video_phrases(
                generated_output_ids,
                tokenizer=tokenizer,
                phrases=phrases,
            )
            if len(phrase_texts) == 0:
                raise ValueError("no phrases found for video inference")

            if video_init_kwargs is None:
                video_init_kwargs = {}
            inference_state = video_seg_engine.init_video(
                video_resource_path,
                **video_init_kwargs,
                )

            if external_query_embeds is None:
                raise ValueError("no external query embeddings found for video inference")
            num_segments = min(len(phrase_texts), external_query_embeds.shape[0])
            video_results = []
            for seg_idx in range(num_segments):
                video_results.append(
                    video_seg_engine.segment_with_phrase_and_queries(
                        phrase=phrase_texts[seg_idx],
                        external_query_embed=external_query_embeds[seg_idx],
                        start_frame=start_frame,
                        inference_state=inference_state,
                        max_frame_num_to_track=max_frame_num_to_track,
                        propagate_both_directions=propagate_both_directions,
                    )
                )

        except Exception as exp:
            print("Video segmentation inference error:", exp)
            video_results = None

        return generated_output_ids, video_results


    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool = True,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ):
        model_inputs = _Qwen3VLForConditionalGeneration.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

        if getattr(self, "SEG_START", None) == "1":
            self.SEG_START = None

        current_input_ids = model_inputs.get("input_ids")
        attention_mask_2d = model_inputs.get("attention_mask")

        if (
            current_input_ids is not None
            and current_input_ids.shape[1] > 0
            and current_input_ids[0, 0] == self.config.ref_end_token_index
            and not getattr(self, "disable_seg_query_generation", False)
        ):
            self.SEG_START = "1"

            ref_end_embedding = self.model.get_input_embeddings()(
                torch.tensor([self.config.ref_end_token_index], dtype=torch.long, device=self.model.device)
            ).unsqueeze(0)
            seg_start_embedding = self.model.get_input_embeddings()(
                torch.tensor([self.config.seg_start_token_index], dtype=torch.long, device=self.model.device)
            ).unsqueeze(0)
            seg_end_embedding = self.model.get_input_embeddings()(
                torch.tensor([self.config.seg_end_token_index], dtype=torch.long, device=self.model.device)
            ).unsqueeze(0)

            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = torch.cat(
                [ref_end_embedding, seg_start_embedding, self.model.mask_queries.unsqueeze(0), seg_end_embedding],
                dim=1,
            )

            if attention_mask_2d is not None and attention_mask_2d.ndim == 2:
                extra_tokens = self.config.max_seg_nums + 2
                attention_mask_2d = torch.cat(
                    [attention_mask_2d, attention_mask_2d.new_ones((attention_mask_2d.shape[0], extra_tokens))],
                    dim=-1,
                )
                model_inputs["attention_mask"] = attention_mask_2d

            cache_position = model_inputs.get("cache_position")
            if cache_position is not None:
                start_position = cache_position[0]
                query_len = model_inputs["inputs_embeds"].shape[1]
                model_inputs["cache_position"] = torch.arange(
                    start_position,
                    start_position + query_len,
                    device=cache_position.device,
                    dtype=cache_position.dtype,
                )

        attention_mask_2d = model_inputs.get("attention_mask")
        if (
            getattr(self, "SEG_START", None) == "1"
            and attention_mask_2d is not None
            and attention_mask_2d.ndim == 2
        ):
            current_input_length = model_inputs["inputs_embeds"].shape[1]
            model_inputs["attention_mask"] = full_attn_mask(
                current_input_length, attention_mask_2d.shape[1], attention_mask_2d
            )

        model_inputs.pop("labels", None)
        return model_inputs

    def _cache_dependant_input_preparation(self, input_ids, inputs_embeds, cache_position):
        if cache_position is None:
            return inputs_embeds, input_ids

        current_input_length = cache_position.shape[0]
        if inputs_embeds is not None:
            if inputs_embeds.shape[1] != current_input_length:
                inputs_embeds = inputs_embeds[:, -current_input_length:, :]
            return inputs_embeds, input_ids

        if input_ids is not None and input_ids.shape[1] != current_input_length:
            input_ids = input_ids[:, -current_input_length:]
        return inputs_embeds, input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        seg_start = self.SEG_START
        if seg_start is not None:
            self.seg_output_embeddings.append(outputs['hidden_states'][-1][:,2:-1]) # except the start and end token 

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens
        )
 
        return model_kwargs
