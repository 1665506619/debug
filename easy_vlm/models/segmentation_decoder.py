import copy
import os
from types import MethodType

import torch
import torch.nn as nn


def _resolve_sam3_checkpoint_path(mask_decoder_model):
    if mask_decoder_model is None:
        return None
    if os.path.isfile(mask_decoder_model):
        return mask_decoder_model
    if os.path.isdir(mask_decoder_model):
        candidate = os.path.join(mask_decoder_model, "sam3.pt")
        if os.path.isfile(candidate):
            return candidate
    return None


def _slice_backbone_out(backbone_out, idx):
    if torch.is_tensor(backbone_out):
        if backbone_out.dim() == 0 or backbone_out.shape[0] <= idx:
            return backbone_out
        return backbone_out[idx : idx + 1]
    if isinstance(backbone_out, tuple):
        return tuple(_slice_backbone_out(item, idx) for item in backbone_out)
    if isinstance(backbone_out, list):
        return [_slice_backbone_out(item, idx) for item in backbone_out]
    if isinstance(backbone_out, dict):
        return {key: _slice_backbone_out(value, idx) for key, value in backbone_out.items()}
    return backbone_out


def _cast_float_tensors(value, dtype):
    if torch.is_tensor(value):
        if torch.is_floating_point(value):
            return value.to(dtype=dtype)
        return value
    if isinstance(value, tuple):
        return tuple(_cast_float_tensors(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_float_tensors(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _cast_float_tensors(item, dtype) for key, item in value.items()}
    return value


def _build_empty_geometric_prompt(prompt_cls, batch_size, device, dtype):
    return prompt_cls(
        box_embeddings=torch.zeros(0, batch_size, 4, device=device, dtype=dtype),
        box_mask=torch.zeros(batch_size, 0, device=device, dtype=torch.bool),
        point_embeddings=torch.zeros(0, batch_size, 2, device=device, dtype=dtype),
        point_mask=torch.zeros(batch_size, 0, device=device, dtype=torch.bool),
    )


def _patch_geometry_encoder_dtype_compat(geometry_encoder):
    def _encode_points(self, points, points_mask, points_labels, img_feats):
        points_embed = None
        n_points, bs = points.shape[:2]
        if n_points == 0:
            empty = torch.zeros(
                0,
                bs,
                self.d_model,
                device=points.device,
                dtype=self.label_embed.weight.dtype,
            )
            return empty, points_mask

        if self.points_direct_project is not None:
            proj = self.points_direct_project(points.to(self.points_direct_project.weight.dtype))
            assert points_embed is None
            points_embed = proj

        if self.points_pool_project is not None:
            grid = points.transpose(0, 1).unsqueeze(2)
            grid = (grid * 2) - 1
            sampled = torch.nn.functional.grid_sample(
                img_feats, grid.to(img_feats.dtype), align_corners=False
            )
            assert list(sampled.shape) == [bs, self.d_model, n_points, 1]
            sampled = sampled.squeeze(-1).permute(2, 0, 1)
            proj = self.points_pool_project(sampled.to(self.points_pool_project.weight.dtype))
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj

        if self.points_pos_enc_project is not None:
            x, y = points.unbind(-1)
            enc_x, enc_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
            enc_x = enc_x.view(n_points, bs, enc_x.shape[-1])
            enc_y = enc_y.view(n_points, bs, enc_y.shape[-1])
            enc = torch.cat([enc_x, enc_y], -1).to(self.points_pos_enc_project.weight.dtype)

            proj = self.points_pos_enc_project(enc)
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj

        type_embed = self.label_embed(points_labels.long())
        return type_embed + points_embed, points_mask

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
        import torchvision
        from .sam3_full.geometry_encoders import box_cxcywh_to_xyxy

        boxes_embed = None
        n_boxes, bs = boxes.shape[:2]
        if n_boxes == 0:
            empty = torch.zeros(
                0,
                bs,
                self.d_model,
                device=boxes.device,
                dtype=self.label_embed.weight.dtype,
            )
            return empty, boxes_mask

        if self.boxes_direct_project is not None:
            proj = self.boxes_direct_project(boxes.to(self.boxes_direct_project.weight.dtype))
            assert boxes_embed is None
            boxes_embed = proj

        if self.boxes_pool_project is not None:
            H, W = img_feats.shape[-2:]
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
            scale = scale.view(1, 1, 4)
            boxes_xyxy = boxes_xyxy * scale
            sampled = torchvision.ops.roi_align(
                img_feats,
                boxes_xyxy.float().transpose(0, 1).unbind(0),
                self.roi_size,
            )
            assert list(sampled.shape) == [bs * n_boxes, self.d_model, self.roi_size, self.roi_size]
            proj = self.boxes_pool_project(sampled.to(self.boxes_pool_project.weight.dtype))
            proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        if self.boxes_pos_enc_project is not None:
            cx, cy, w, h = boxes.unbind(-1)
            enc = self.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
            enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1]).to(
                self.boxes_pos_enc_project.weight.dtype
            )

            proj = self.boxes_pos_enc_project(enc)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        type_embed = self.label_embed(boxes_labels.long())
        return type_embed + boxes_embed, boxes_mask

    geometry_encoder._encode_points = MethodType(_encode_points, geometry_encoder)
    geometry_encoder._encode_boxes = MethodType(_encode_boxes, geometry_encoder)
    return geometry_encoder


def _create_sam3_full_transformer(num_queries):
    from .sam3_full import builders as sam3_builder
    from .sam3_full.decoder import TransformerDecoder, TransformerDecoderLayer
    from .sam3_full.model_misc import MultiheadAttentionWrapper as MultiheadAttention
    from .sam3_full.model_misc import TransformerWrapper

    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive, got {num_queries}")

    encoder = sam3_builder._create_transformer_encoder()

    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=num_queries,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _load_sam3_full_checkpoint(model, checkpoint_path):
    from iopath.common.file_io import g_pathmgr

    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    for key in [
        "transformer.decoder.query_embed.weight",
        "transformer.decoder.reference_points.weight",
    ]:
        sam3_image_ckpt.pop(key, None)

    model.load_state_dict(sam3_image_ckpt, strict=False)


def _build_sam3_full_image_model(checkpoint_path, training, num_queries):
    from .sam3_full import builders as sam3_builder

    from .sam3_full.sam3_image import Sam3Image

    bpe_path = sam3_builder.resolve_sam3_bpe_path()

    vision_encoder = sam3_builder._create_vision_backbone(
        compile_mode=None,
        enable_inst_interactivity=False,
    )
    text_encoder = sam3_builder._create_text_encoder(bpe_path)
    backbone = sam3_builder._create_vl_backbone(vision_encoder, text_encoder)
    transformer = _create_sam3_full_transformer(num_queries)
    dot_prod_scoring = sam3_builder._create_dot_product_scoring()
    segmentation_head = sam3_builder._create_segmentation_head(compile_mode=None)
    input_geometry_encoder = sam3_builder._create_geometry_encoder()
    input_geometry_encoder = _patch_geometry_encoder_dtype_compat(input_geometry_encoder)

    model = Sam3Image(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=None,
        matcher=None,
    )

    if checkpoint_path is not None:
        _load_sam3_full_checkpoint(model, checkpoint_path)

    if training:
        model.train()
    else:
        model.eval()
    return model


class SegmentationDecoder(nn.Module):
    def __init__(self, config):
        super(SegmentationDecoder, self).__init__()
        self.config = config
        if config.seg_encoder == "sam3" and config.seg_decoder == "sam3":
            checkpoint_path = _resolve_sam3_checkpoint_path(
                getattr(self.config, "mask_decoder_model", None)
            )
            self.model = _build_sam3_full_image_model(
                checkpoint_path=checkpoint_path,
                training=self.training,
                num_queries=self.config.max_seg_nums,
            )
        else:
            raise NotImplementedError

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "model"):
            self.model.train(mode)
        return self

    def load_model(self, config):
        checkpoint_path = _resolve_sam3_checkpoint_path(
            getattr(self.config, "mask_decoder_model", None)
        )
        self.model = _build_sam3_full_image_model(
            checkpoint_path=checkpoint_path,
            training=self.training,
            num_queries=config.max_seg_nums,
        )

    def get_sam_model(self):
        return self.model

    def get_num_queries(self):
        return self.model.transformer.decoder.num_queries

    def get_sam_vision_encoder(self):
        return self.model.backbone.vision_backbone

    def encoder(self, pixel_values):
        vision_dtype = next(self.model.backbone.vision_backbone.parameters()).dtype
        transformer_dtype = next(self.model.transformer.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)
        backbone_out = self.model.backbone.forward_image(pixel_values)
        backbone_out = _cast_float_tensors(backbone_out, transformer_dtype)
        return {"backbone_out": backbone_out}

    def select_vision_outputs(self, vision_outputs_batch, idx: int):
        return {"backbone_out": _slice_backbone_out(vision_outputs_batch["backbone_out"], idx)}

    def decoder(self, vision_outputs, text_embeds, text_attn_mask, query_embed):
        from .sam3_full.data_misc import FindStage
        from .sam3_full.geometry_encoders import Prompt

        if text_embeds.dim() != 3:
            raise ValueError(
                f"text_embeds must be [num_prompts, seq_len, dim], got shape {tuple(text_embeds.shape)}"
            )
        if text_attn_mask.dim() != 2:
            raise ValueError(
                f"text_attn_mask must be [num_prompts, seq_len], got shape {tuple(text_attn_mask.shape)}"
            )
        if query_embed.dim() != 3:
            raise ValueError(
                f"query_embed must be [num_prompts, num_queries, dim], got shape {tuple(query_embed.shape)}"
            )
        expected_num_queries = self.get_num_queries()
        if query_embed.shape[1] > expected_num_queries:
            raise ValueError(
                "sam3_full cannot consume more external Qwen queries than its learned decoder slots: "
                f"got {query_embed.shape[1]}, max supported {expected_num_queries}."
            )

        num_prompts = text_embeds.shape[0]
        device = text_embeds.device
        prompt_dtype = next(self.model.geometry_encoder.parameters()).dtype
        transformer_dtype = next(self.model.transformer.parameters()).dtype

        backbone_out = _cast_float_tensors(copy.copy(vision_outputs["backbone_out"]), transformer_dtype)
        backbone_out["language_features"] = text_embeds.permute(1, 0, 2).to(transformer_dtype)
        backbone_out["language_mask"] = ~text_attn_mask.bool()

        find_input = FindStage(
            img_ids=torch.zeros(num_prompts, device=device, dtype=torch.long),
            text_ids=torch.arange(num_prompts, device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geometric_prompt = _build_empty_geometric_prompt(
            Prompt,
            num_prompts,
            device,
            prompt_dtype,
        )

        prompt, prompt_mask, backbone_out = self.model._encode_prompt(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
        )
        backbone_out, encoder_out, _ = self.model._run_encoder(
            backbone_out=backbone_out,
            find_input=find_input,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )

        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        out, hs = self.model._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
            external_query_embed=query_embed.permute(1, 0, 2),
        )
        self.model._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=find_input.img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
        )

        if out.get("pred_logits") is not None and out["pred_logits"].dim() == 3 and out["pred_logits"].shape[-1] == 1:
            out["pred_logits"] = out["pred_logits"].squeeze(-1)
        return out
