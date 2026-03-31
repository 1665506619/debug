from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .sam3_native_video_loss import Sam3NativeVideoLossAdapter
from .segmentation_decoder import _resolve_sam3_checkpoint_path
from .sam3_full.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity


class VideoSegTrainer(nn.Module):
    def __init__(
        self,
        sam3_video_model: nn.Module,
        sam3_native_video_loss: Optional[Sam3NativeVideoLossAdapter] = None,
    ):
        super().__init__()
        self.sam3_video_model = sam3_video_model
        self.sam3_native_video_loss = sam3_native_video_loss

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

    @contextmanager
    def _single_rank_video_forward(self):
        """Run the internal SAM3 video stack in per-rank local mode under outer DDP."""
        sam3_video_model = self.sam3_video_model
        detector = sam3_video_model.detector
        orig_rank = sam3_video_model.rank
        orig_world_size = sam3_video_model.world_size
        orig_detector_rank = detector.rank
        orig_detector_world_size = detector.world_size
        sam3_video_model.rank = detector.rank = 0
        sam3_video_model.world_size = detector.world_size = 1
        try:
            yield
        finally:
            sam3_video_model.rank = orig_rank
            sam3_video_model.world_size = orig_world_size
            detector.rank = orig_detector_rank
            detector.world_size = orig_detector_world_size

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

        with self._single_rank_video_forward():
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
        }

    def forward_video_with_phrase_and_queries_for_loss(
        self,
        phrase: str,
        external_query_embed: torch.Tensor,
        video_masks: torch.Tensor,
        video_mask_valid: torch.Tensor,
        object_ids: Optional[torch.Tensor] = None,
        inference_state: Optional[Dict[str, Any]] = None,
        resource_path=None,
        frames: Optional[Sequence[Any]] = None,
        start_frame: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if self.sam3_native_video_loss is None:
            raise RuntimeError("sam3_native_video_loss is not initialized")
        if not phrase:
            raise ValueError("phrase must be a non-empty string")
        if external_query_embed is None:
            raise ValueError("external_query_embed must not be None")
        if external_query_embed.dim() != 2:
            raise ValueError(
                "external_query_embed must be a 2D tensor shaped [num_queries, dim]"
            )
        if video_masks is None or video_mask_valid is None:
            raise ValueError("video_masks and video_mask_valid must not be None")
        if video_masks.dim() != 4:
            raise ValueError(
                f"video_masks must have shape [N,T,H,W], got {tuple(video_masks.shape)}"
            )
        if video_mask_valid.dim() != 2:
            raise ValueError(
                "video_mask_valid must have shape [N,T], got "
                f"{tuple(video_mask_valid.shape)}"
            )
        if video_masks.shape[:2] != video_mask_valid.shape:
            raise ValueError(
                "video_masks and video_mask_valid leading dims must match, got "
                f"{tuple(video_masks.shape[:2])} vs {tuple(video_mask_valid.shape)}"
            )
        video_mask_valid = video_mask_valid.to(dtype=torch.bool)

        if inference_state is None:
            init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
            inference_state = self.init_video(
                resource_path=resource_path,
                frames=frames,
                **init_kwargs,
            )
        if inference_state["num_frames"] != video_masks.shape[1]:
            raise ValueError(
                "video_masks frame count does not match initialized video state, got "
                f"{video_masks.shape[1]} vs {inference_state['num_frames']}"
            )

        if object_ids is not None:
            object_ids = object_ids.to(device=video_masks.device, dtype=torch.long)
            if object_ids.numel() != video_masks.shape[0]:
                raise ValueError(
                    "object_ids and video_masks must have the same instance count, got "
                    f"{object_ids.numel()} vs {video_masks.shape[0]}"
                )

        model_dtype = next(self.sam3_video_model.parameters()).dtype
        query_embed = external_query_embed.to(device=self.device, dtype=model_dtype)
        inference_state["constants"]["external_query_embed"] = query_embed
        start_frame = self._resolve_start_frame(video_mask_valid, start_frame)

        supervised_frames = torch.nonzero(
            video_mask_valid.any(dim=0),
            as_tuple=False,
        ).flatten()
        raw_frame_outputs: List[Dict[str, Any]] = []
        gt_masks_per_frame: List[torch.Tensor] = []
        object_ids_per_frame: List[Optional[torch.Tensor]] = []
        supervised_frame_indices: List[int] = []

        with self._single_rank_video_forward():
            frame_idx, prompt_out = self.sam3_video_model.add_prompt_for_training(
                inference_state,
                frame_idx=start_frame,
                text_str=phrase,
            )
            if video_mask_valid[:, frame_idx].any():
                gt_valid = video_mask_valid[:, frame_idx]
                raw_frame_outputs.append(prompt_out)
                gt_masks_per_frame.append(video_masks[:, frame_idx][gt_valid])
                object_ids_per_frame.append(
                    None if object_ids is None else object_ids[gt_valid]
                )
                supervised_frame_indices.append(frame_idx)

            for out_frame_idx, out in self.sam3_video_model.propagate_in_video_for_training(
                inference_state,
                start_frame_idx=start_frame,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=False,
                include_start_frame=False,
            ):
                if not video_mask_valid[:, out_frame_idx].any():
                    continue
                gt_valid = video_mask_valid[:, out_frame_idx]
                raw_frame_outputs.append(out)
                gt_masks_per_frame.append(video_masks[:, out_frame_idx][gt_valid])
                object_ids_per_frame.append(
                    None if object_ids is None else object_ids[gt_valid]
                )
                supervised_frame_indices.append(out_frame_idx)

        return self.sam3_native_video_loss.build_payload(
            raw_frame_outputs=raw_frame_outputs,
            gt_masks_per_frame=gt_masks_per_frame,
            object_ids_per_frame=object_ids_per_frame,
            supervised_frame_indices=supervised_frame_indices,
        )

    def compute_sam3_native_video_loss(self, **kwargs):
        payload = self.forward_video_with_phrase_and_queries_for_loss(**kwargs)
        loss_dict = self.sam3_native_video_loss.compute_loss(payload)
        return {
            "payload": payload,
            "loss_dict": loss_dict,
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

        # RoPE caches are runtime-recomputable buffers and are no longer persisted
        # in the local vitdet implementation. Old checkpoints may still contain them.
        ckpt = {
            k: v
            for k, v in ckpt.items()
            if not k.endswith(".attn.freqs_cis")
        }

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
    sam3_native_video_loss = Sam3NativeVideoLossAdapter(config)
    trainer = VideoSegTrainer(
        sam3_video_model,
        sam3_native_video_loss=sam3_native_video_loss,
    )
    trainer.train()
    return trainer
