from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .segmentation_decoder import _resolve_sam3_checkpoint_path
from .sam3_full.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity


class VideoSegTrainer(nn.Module):
    def __init__(self, sam3_video_model: nn.Module):
        super().__init__()
        self.sam3_video_model = sam3_video_model

    def train(self, mode: bool = True):
        super().train(mode)
        self.sam3_video_model.train(mode)
        if mode:
            # Keep SAM3's internal detector/tracker away from their native training
            # losses and train-only codepaths; we only need differentiable forward.
            self.sam3_video_model.detector.eval()
            self.sam3_video_model.tracker.eval()
        return self

    @property
    def device(self):
        return next(self.sam3_video_model.parameters()).device

    def init_video(
        self,
        resource_path=None,
        frames: Optional[Sequence[Any]] = None,
        **kwargs,
    ):
        return self.sam3_video_model.init_state_for_training(
            resource_path=resource_path,
            frames=frames,
            **kwargs,
        )

    def _resolve_start_frame(
        self,
        video_mask_valid: Optional[torch.Tensor],
        preferred_start_frame: Optional[int] = None,
    ) -> int:
        if video_mask_valid is None or video_mask_valid.numel() == 0:
            return 0 if preferred_start_frame is None else int(preferred_start_frame)

        num_frames = video_mask_valid.shape[-1]
        if preferred_start_frame is not None:
            preferred_start_frame = int(preferred_start_frame)
            if 0 <= preferred_start_frame < num_frames and video_mask_valid[
                :, preferred_start_frame
            ].any():
                return preferred_start_frame

        supervised_frames = torch.nonzero(
            video_mask_valid.any(dim=0),
            as_tuple=False,
        ).flatten()
        if supervised_frames.numel() == 0:
            return 0
        return int(supervised_frames[0].item())

    def segment_with_phrase_and_queries(
        self,
        phrase: str,
        external_query_embed: torch.Tensor,
        inference_state: Optional[Dict[str, Any]] = None,
        resource_path=None,
        frames: Optional[Sequence[Any]] = None,
        video_mask_valid: Optional[torch.Tensor] = None,
        start_frame: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if inference_state is None:
            init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
            inference_state = self.init_video(
                resource_path=resource_path,
                frames=frames,
                **init_kwargs,
            )

        if not phrase:
            raise ValueError("phrase must be a non-empty string")
        if external_query_embed is None:
            raise ValueError("external_query_embed must not be None")
        if external_query_embed.dim() != 2:
            raise ValueError(
                "external_query_embed must be a 2D tensor shaped [num_queries, dim]"
            )

        model_dtype = next(self.sam3_video_model.parameters()).dtype
        query_embed = external_query_embed.to(device=self.device, dtype=model_dtype)
        inference_state["constants"]["external_query_embed"] = query_embed

        start_frame = self._resolve_start_frame(video_mask_valid, start_frame)

        frame_idx, prompt_out = self.sam3_video_model.add_prompt_for_training(
            inference_state,
            frame_idx=start_frame,
            text_str=phrase,
        )

        num_frames = inference_state["num_frames"]
        frame_outputs: List[Optional[Dict[str, Any]]] = [None] * num_frames
        frame_outputs[frame_idx] = prompt_out

        for out_frame_idx, out in self.sam3_video_model.propagate_in_video_for_training(
            inference_state,
            start_frame_idx=start_frame,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
            include_start_frame=False,
        ):
            frame_outputs[out_frame_idx] = out

        return {
            "phrase": phrase,
            "start_frame": start_frame,
            "num_frames": num_frames,
            "frame_outputs": frame_outputs,
            "pred_masks_per_frame": [
                None if out is None else out["pred_mask_logits"] for out in frame_outputs
            ],
            "pred_scores_per_frame": [
                None if out is None else out["out_probs"] for out in frame_outputs
            ],
            "tracked_obj_ids": [
                None if out is None else out["out_obj_ids"] for out in frame_outputs
            ],
            "inference_state": inference_state,
        }


def build_video_seg_trainer(config):
    from iopath.common.file_io import g_pathmgr
    from .sam3_full.builders import (
        _create_geometry_encoder,
        _create_segmentation_head,
        _create_sam3_transformer,
        _create_text_encoder,
        _create_vision_backbone,
        resolve_sam3_bpe_path,
        build_tracker,
    )
    from .sam3_full.model_misc import DotProductScoring, MLP
    from .sam3_full.sam3_image import Sam3ImageOnVideoMultiGPU
    from .sam3_full.vl_combiner import SAM3VLBackbone

    checkpoint_path = _resolve_sam3_checkpoint_path(
        getattr(config, "mask_decoder_model", None)
    )
    bpe_path = resolve_sam3_bpe_path()
    tracker = build_tracker(apply_temporal_disambiguation=True)

    visual_neck = _create_vision_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=True)
    segmentation_head = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256,
        d_proj=256,
        prompt_mlp=main_dot_prod_mlp,
    )

    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=True,
    )
    sam3_video_model = Sam3VideoInferenceWithInstanceInteractivity(
        detector=detector,
        tracker=tracker,
        score_threshold_detection=0.5,
        assoc_iou_thresh=0.1,
        det_nms_thresh=0.1,
        new_det_thresh=0.7,
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=True,
        min_trk_keep_alive=-1,
        max_trk_keep_alive=30,
        init_trk_keep_alive=30,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        recondition_every_nth_frame=16,
        masklet_confirmation_enable=False,
        decrease_trk_keep_alive_for_empty_masklets=False,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=False,
    )

    if checkpoint_path is not None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        missing_keys, unexpected_keys = sam3_video_model.load_state_dict(
            ckpt,
            strict=True,
        )
        if missing_keys:
            raise RuntimeError(f"Missing keys when loading SAM3 video checkpoint: {missing_keys}")
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys when loading SAM3 video checkpoint: {unexpected_keys}"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam3_video_model.to(device=device)
    trainer = VideoSegTrainer(sam3_video_model)
    trainer.train()
    return trainer
