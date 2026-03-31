import torch


def masks_to_boxes(masks: torch.Tensor, obj_ids: list[int]):
    with torch.autograd.profiler.record_function("perflib: masks_to_boxes"):
        assert masks.shape[0] == len(obj_ids)
        assert masks.dim() == 3

        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        n, h, w = masks.shape
        device = masks.device
        y = torch.arange(h, device=device).view(1, h)
        x = torch.arange(w, device=device).view(1, w)

        masks_with_obj = masks != 0
        masks_with_obj_x = masks_with_obj.amax(dim=1)
        masks_with_obj_y = masks_with_obj.amax(dim=2)
        masks_without_obj_x = ~masks_with_obj_x
        masks_without_obj_y = ~masks_with_obj_y

        x0 = torch.amin((masks_without_obj_x * w) + (masks_with_obj_x * x), dim=1)
        y0 = torch.amin((masks_without_obj_y * h) + (masks_with_obj_y * y), dim=1)
        x1 = torch.amax(masks_with_obj_x * x, dim=1)
        y1 = torch.amax(masks_with_obj_y * y, dim=1)

        return torch.stack([x0, y0, x1, y1], dim=1).to(dtype=torch.float)


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    n, h, w = pred_masks.shape
    m, _, _ = gt_masks.shape
    pred_flat = pred_masks.view(n, 1, h * w)
    gt_flat = gt_masks.view(1, m, h * w)
    intersection = (pred_flat & gt_flat).sum(dim=2).float()
    union = (pred_flat | gt_flat).sum(dim=2).float()
    return intersection / union.clamp(min=1)
