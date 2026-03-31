from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from .sam3_full.box_ops import box_xyxy_to_cxcywh
from .sam3_full.model_misc import SAM3Output
from .sam3_full.train_utils.loss.loss_fns import Boxes, Masks
from .sam3_full.train_utils.loss.sam3_loss import Sam3LossWrapper
from .sam3_full.train_utils.matcher import BinaryHungarianMatcherV2


@dataclass
class Sam3NativeVideoLossPayload:
    find_stages: SAM3Output
    find_targets: List[Dict[str, torch.Tensor]]
    supervised_frame_indices: List[int]


class Sam3NativeVideoLossAdapter(nn.Module):
    """
    Current scope:
    - native Masks loss
    - optional native Boxes loss when frame outputs carry real detector boxes

    Not wired in this adapter yet:
    - semantic segmentation loss
    - det/trk association losses
    """

    _OPTIONAL_OUTPUT_KEYS = ("frame_idx", "is_video_grounding_batch", "Q_det")

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.use_box_loss = bool(
            getattr(config, "sam3_native_video_use_box_loss", False)
        )

        loss_fns_find: List[nn.Module] = [
            Masks(
                weight_dict={
                    "loss_mask": float(
                        getattr(
                            config,
                            "sam3_native_video_mask_loss_weight",
                            config.bce_loss_weight,
                        )
                    ),
                    "loss_dice": float(
                        getattr(
                            config,
                            "sam3_native_video_dice_loss_weight",
                            config.dice_loss_weight,
                        )
                    ),
                },
                compute_aux=False,
                apply_loss_to_det_queries_in_video_grounding=True,
            )
        ]
        if self.use_box_loss:
            loss_fns_find.append(
                Boxes(
                    weight_dict={
                        "loss_bbox": float(
                            getattr(config, "sam3_native_video_box_loss_weight", 0.0)
                        ),
                        "loss_giou": float(
                            getattr(config, "sam3_native_video_giou_loss_weight", 0.0)
                        ),
                    },
                    compute_aux=False,
                    apply_loss_to_det_queries_in_video_grounding=True,
                )
            )

        self.matcher = BinaryHungarianMatcherV2(
            cost_class=float(
                getattr(
                    config,
                    "sam3_native_video_matcher_cost_class",
                    max(float(getattr(config, "cls_loss_weight", 0.5)), 1e-3),
                )
            ),
            cost_bbox=(
                float(getattr(config, "sam3_native_video_matcher_cost_bbox", 1.0))
                if self.use_box_loss
                else 0.0
            ),
            cost_giou=(
                float(getattr(config, "sam3_native_video_matcher_cost_giou", 1.0))
                if self.use_box_loss
                else 0.0
            ),
            focal=False,
            remove_samples_with_0_gt=True,
        )
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns_find,
            normalization="local",
            matcher=self.matcher,
            o2m_matcher=None,
            normalize_by_stage_num=bool(
                getattr(config, "sam3_native_video_normalize_by_stage_num", True)
            ),
        )

    @staticmethod
    def _normalize_xyxy_boxes(
        boxes_xyxy: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy.reshape(-1, 4)
        boxes_xyxy = boxes_xyxy.to(dtype=torch.float32).clone()
        boxes_xyxy[:, [0, 2]] /= max(float(width), 1.0)
        boxes_xyxy[:, [1, 3]] /= max(float(height), 1.0)
        return boxes_xyxy.clamp_(0.0, 1.0)

    @staticmethod
    def _ensure_bool_masks(masks: torch.Tensor) -> torch.Tensor:
        return masks if masks.dtype == torch.bool else masks > 0

    @staticmethod
    def _masks_to_boxes_safe(masks: torch.Tensor) -> torch.Tensor:
        masks = masks.bool()
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float32)

        _, height, width = masks.shape
        y = torch.arange(height, dtype=torch.float32, device=masks.device)
        x = torch.arange(width, dtype=torch.float32, device=masks.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        x_mask = masks * xx.unsqueeze(0)
        y_mask = masks * yy.unsqueeze(0)
        valid = masks.flatten(1).any(dim=1)
        x_min = x_mask.masked_fill(~masks, float("inf")).flatten(1).min(dim=1).values
        y_min = y_mask.masked_fill(~masks, float("inf")).flatten(1).min(dim=1).values
        x_max = x_mask.flatten(1).max(dim=1).values + 1.0
        y_max = y_mask.flatten(1).max(dim=1).values + 1.0

        boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        boxes = boxes.masked_fill(~valid[:, None], 0.0)
        return boxes

    def _build_output_from_raw_frame(self, raw_frame_out: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = ("pred_logits", "pred_masks")
        missing_keys = [key for key in required_keys if key not in raw_frame_out]
        if missing_keys:
            raise KeyError(
                "raw_frame_out is missing required native SAM3 output keys: "
                + ", ".join(missing_keys)
            )

        pred_logits = raw_frame_out["pred_logits"]
        pred_masks = raw_frame_out["pred_masks"]
        if pred_logits.dim() != 3 or pred_logits.shape[0] != 1 or pred_logits.shape[-1] != 1:
            raise ValueError(
                "Expected pred_logits to have shape [1, Q, 1], got "
                f"{tuple(pred_logits.shape)}"
            )
        if pred_masks.dim() != 4 or pred_masks.shape[0] != 1:
            raise ValueError(
                "Expected pred_masks to have shape [1, Q, H, W], got "
                f"{tuple(pred_masks.shape)}"
            )
        if pred_logits.shape[1] != pred_masks.shape[1]:
            raise ValueError(
                "Mismatch between logits and masks query dims: "
                f"{pred_logits.shape[1]} vs {pred_masks.shape[1]}"
            )

        output_dict = {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
        }

        has_pred_boxes = "pred_boxes" in raw_frame_out or "pred_boxes_xyxy" in raw_frame_out
        if has_pred_boxes:
            if "pred_boxes" not in raw_frame_out or "pred_boxes_xyxy" not in raw_frame_out:
                raise KeyError(
                    "raw_frame_out must provide both pred_boxes and pred_boxes_xyxy together"
                )
            pred_boxes = raw_frame_out["pred_boxes"]
            pred_boxes_xyxy = raw_frame_out["pred_boxes_xyxy"]
            if pred_boxes.dim() != 3 or pred_boxes.shape != (1, pred_logits.shape[1], 4):
                raise ValueError(
                    "Expected pred_boxes to have shape [1, Q, 4], got "
                    f"{tuple(pred_boxes.shape)}"
                )
            if pred_boxes_xyxy.dim() != 3 or pred_boxes_xyxy.shape != (1, pred_logits.shape[1], 4):
                raise ValueError(
                    "Expected pred_boxes_xyxy to have shape [1, Q, 4], got "
                    f"{tuple(pred_boxes_xyxy.shape)}"
                )
            output_dict["pred_boxes"] = pred_boxes
            output_dict["pred_boxes_xyxy"] = pred_boxes_xyxy
        elif self.use_box_loss:
            raise ValueError(
                "sam3_native_video_use_box_loss=True but raw_frame_out does not carry "
                "real pred_boxes/pred_boxes_xyxy"
            )

        for key in self._OPTIONAL_OUTPUT_KEYS:
            if key in raw_frame_out and raw_frame_out[key] is not None:
                output_dict[key] = raw_frame_out[key]
        return output_dict

    def _build_target_for_frame(
        self,
        gt_masks: torch.Tensor,
        object_ids: Optional[torch.Tensor] = None,
        is_exhaustive: bool = True,
    ) -> Dict[str, torch.Tensor]:
        gt_masks = self._ensure_bool_masks(gt_masks)
        if gt_masks.dim() != 3:
            raise ValueError(
                f"Expected gt_masks to have shape [N, H, W], got {tuple(gt_masks.shape)}"
            )

        device = gt_masks.device
        num_instances = gt_masks.shape[0]
        height, width = gt_masks.shape[-2:]
        if object_ids is None:
            object_ids = torch.arange(num_instances, device=device, dtype=torch.long)
        else:
            object_ids = object_ids.to(device=device, dtype=torch.long)
            if object_ids.numel() != num_instances:
                raise ValueError(
                    "object_ids and gt_masks must have the same length, got "
                    f"{object_ids.numel()} vs {num_instances}"
                )

        is_valid_mask = gt_masks.flatten(1).any(dim=1)
        boxes_xyxy = self._normalize_xyxy_boxes(
            self._masks_to_boxes_safe(gt_masks),
            height,
            width,
        )
        boxes = box_xyxy_to_cxcywh(boxes_xyxy)

        return {
            "boxes": boxes,
            "boxes_xyxy": boxes_xyxy,
            "boxes_padded": boxes.unsqueeze(0),
            "positive_map": torch.ones(num_instances, 1, device=device, dtype=boxes.dtype),
            "num_boxes": torch.tensor([num_instances], device=device, dtype=torch.long),
            "masks": gt_masks,
            "semantic_masks": None,
            "is_valid_mask": is_valid_mask,
            "is_exhaustive": torch.tensor([is_exhaustive], device=device, dtype=torch.bool),
            "object_ids_packed": object_ids,
            "object_ids_padded": object_ids.unsqueeze(0),
        }

    def build_payload(
        self,
        *,
        raw_frame_outputs: Sequence[Dict[str, Any]],
        gt_masks_per_frame: Sequence[torch.Tensor],
        object_ids_per_frame: Optional[Sequence[Optional[torch.Tensor]]] = None,
        supervised_frame_indices: Optional[Iterable[int]] = None,
    ) -> Sam3NativeVideoLossPayload:
        if len(raw_frame_outputs) != len(gt_masks_per_frame):
            raise ValueError(
                "raw_frame_outputs and gt_masks_per_frame must have the same length, got "
                f"{len(raw_frame_outputs)} vs {len(gt_masks_per_frame)}"
            )
        if object_ids_per_frame is None:
            object_ids_per_frame = [None] * len(raw_frame_outputs)
        elif len(object_ids_per_frame) != len(raw_frame_outputs):
            raise ValueError(
                "object_ids_per_frame and raw_frame_outputs must have the same length, got "
                f"{len(object_ids_per_frame)} vs {len(raw_frame_outputs)}"
            )

        if supervised_frame_indices is None:
            supervised_frame_indices = list(range(len(raw_frame_outputs)))
        else:
            supervised_frame_indices = list(supervised_frame_indices)
            if len(supervised_frame_indices) != len(raw_frame_outputs):
                raise ValueError(
                    "supervised_frame_indices and raw_frame_outputs must have the same "
                    f"length, got {len(supervised_frame_indices)} vs {len(raw_frame_outputs)}"
                )

        stages = SAM3Output()
        targets: List[Dict[str, torch.Tensor]] = []
        for raw_frame_out, gt_masks, object_ids in zip(
            raw_frame_outputs, gt_masks_per_frame, object_ids_per_frame
        ):
            output_dict = self._build_output_from_raw_frame(raw_frame_out)
            target_dict = self._build_target_for_frame(
                gt_masks=gt_masks,
                object_ids=object_ids,
            )
            output_dict["indices"] = self.matcher(output_dict, target_dict)
            stages.append([output_dict])
            targets.append(target_dict)

        return Sam3NativeVideoLossPayload(
            find_stages=stages,
            find_targets=targets,
            supervised_frame_indices=supervised_frame_indices,
        )

    def compute_loss(self, payload: Sam3NativeVideoLossPayload) -> Dict[str, torch.Tensor]:
        loss_dict = self.loss_wrapper(payload.find_stages, payload.find_targets)
        for key, value in loss_dict.items():
            if torch.is_tensor(value) and value.dtype.is_floating_point:
                if not torch.isfinite(value).all():
                    raise RuntimeError(
                        "Non-finite SAM3 native video loss detected for "
                        f"{key}: {value.detach().cpu()}"
                    )
        return loss_dict
