from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class VideoSegEngine:
    def __init__(self, sam3_video_inference):
        self.sam3_video_inference = sam3_video_inference
        self.inference_state = None

    def init_video(self, resource_path, **kwargs):
        self.inference_state = self.sam3_video_inference.init_state(
            resource_path=resource_path,
            **kwargs,
        )
        return self.inference_state

    def segment_with_phrase_and_queries(
        self,
        phrase: str,
        external_query_embed: torch.Tensor,
        start_frame: int = 0,
        inference_state: Optional[Dict[str, Any]] = None,
        max_frame_num_to_track: Optional[int] = None,
        propagate_both_directions: bool = False,
    ):
        if inference_state is None:
            inference_state = self.inference_state
        if inference_state is None:
            raise ValueError(
                "VideoSegEngine has no initialized video state. Call init_video(...) first."
        )
        if not phrase:
            raise ValueError("phrase must be a non-empty string")
        if external_query_embed is None:
            raise ValueError("external_query_embed must not be None")

        query_embed = external_query_embed.detach()
        if query_embed.dim() != 2:
            raise ValueError(
                "external_query_embed must be a 2D tensor shaped [num_queries, dim]"
            )

        query_embed = query_embed.to(
            device=self.sam3_video_inference.device,
            dtype=next(self.sam3_video_inference.parameters()).dtype,
        )
        inference_state["constants"]["external_query_embed"] = query_embed

        frame_idx, prompt_out = self.sam3_video_inference.add_prompt(
            inference_state,
            frame_idx=start_frame,
            text_str=phrase,
        )

        num_frames = inference_state["num_frames"]
        frame_outputs = [None] * num_frames
        frame_outputs[frame_idx] = prompt_out

        for out_frame_idx, out in self.sam3_video_inference.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
        ):
            frame_outputs[out_frame_idx] = out

        if propagate_both_directions:
            for out_frame_idx, out in self.sam3_video_inference.propagate_in_video(
                inference_state,
                start_frame_idx=start_frame,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=True,
            ):
                frame_outputs[out_frame_idx] = out

        return {
            "phrase": phrase,
            "start_frame": start_frame,
            "num_frames": num_frames,
            "frame_outputs": frame_outputs,
            "masks": [
                None if out is None else out["out_binary_masks"] for out in frame_outputs
            ],
            "scores": [
                None if out is None else out["out_probs"] for out in frame_outputs
            ],
            "boxes": [
                None if out is None else out["out_boxes_xywh"] for out in frame_outputs
            ],
            "obj_ids": [
                None if out is None else out["out_obj_ids"] for out in frame_outputs
            ],
            "inference_state": inference_state,
        }
