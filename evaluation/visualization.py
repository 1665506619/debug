import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from pycocotools import mask as maskUtils
from tqdm import tqdm

def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        # COCO polygon format (one instance, possibly multiple parts)
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann, dict) and isinstance(mask_ann.get('counts', None), list):
        # uncompressed RLE dict
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # already compressed RLE dict (counts is bytes/str) or other supported
        rle = mask_ann
    mask = maskUtils.decode(rle)  # (H,W) uint8 {0,1}
    return mask

def annAnyToMasks(mask_ann, h, w):
    """
    mask_ann can be:
      - None
      - dict RLE
      - list of polygons (one instance)
      - list of instances (each is dict/list)
    return: list of (H,W) uint8 masks
    """
    if mask_ann is None:
        return []

    if isinstance(mask_ann, list):
        if len(mask_ann) == 0:
            return []
        # If first element is number => it's a polygon list for ONE instance
        if isinstance(mask_ann[0], (int, float)):
            return [annToMask(mask_ann, h, w)]
        # Otherwise list of anns (instances)
        masks = []
        for m in mask_ann:
            if m is None:
                continue
            masks.append(annToMask(m, h, w))
        return masks

    return [annToMask(mask_ann, h, w)]

def merge_masks(masks, h, w):
    """Union multiple binary masks into one."""
    if len(masks) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    out = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        out |= (m > 0).astype(np.uint8)
    return out

def resize_mask_to_image(mask, target_h, target_w):
    """Nearest-neighbor resize for binary/label masks."""
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask
    pil = Image.fromarray(mask.astype(np.uint8) * 255)  # to 0/255 for PIL
    pil = pil.resize((target_w, target_h), resample=Image.NEAREST)
    out = (np.array(pil) > 127).astype(np.uint8)  # back to {0,1}
    return out

def random_distinct_colors(n, seed=42):
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    colors = np.clip(colors, 60, 255).astype(np.uint8)
    return colors

def masks_to_color(masks, colors=None):
    if len(masks) == 0:
        return None
    h, w = masks[0].shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    if colors is None:
        colors = random_distinct_colors(len(masks), seed=42)
    for i, m in enumerate(masks):
        m = (m > 0)
        color_img[m] = colors[i]  # overlap: later overwrites earlier
    return color_img

def overlay(img, mask_color, alpha=0.5):
    img_f = img.astype(np.float32)
    mask_f = mask_color.astype(np.float32)
    return (img_f * (1 - alpha) + mask_f * alpha).astype(np.uint8)

def visualize_masks(img_path, gt_mask, pred_masks, iou, caption, save_path='vis_result.png', alpha=0.5):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    H, W = img_np.shape[:2]

    # resize masks to image size
    gt_mask = resize_mask_to_image(gt_mask, H, W)
    pred_masks = [resize_mask_to_image(m, H, W) for m in pred_masks]

    # GT red
    gt_color = np.zeros((H, W, 3), dtype=np.uint8)
    gt_color[gt_mask > 0] = [255, 0, 0]

    # Pred multi-color
    pred_color = masks_to_color(pred_masks)
    if pred_color is None:
        pred_color = np.zeros_like(img_np, dtype=np.uint8)

    overlay_pred = overlay(img_np, pred_color, alpha=alpha)

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs[0, 0].imshow(img_np);      axs[0, 0].set_title('Image')
    axs[0, 1].imshow(gt_color);    axs[0, 1].set_title('GT Mask (red)')
    axs[1, 0].imshow(pred_color);  axs[1, 0].set_title(f'Pred Masks Colored (N={len(pred_masks)})')
    axs[1, 1].imshow(overlay_pred);axs[1, 1].set_title('Overlay Pred Masks')

    for ax_row in axs:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.07, 1, 0.92])
    plt.suptitle("IoU: N/A" if iou is None else f"IoU: {iou:.4f}", fontsize=18, color='blue')
    fig.text(0.5, 0.025, caption, wrap=True, fontsize=14, ha='center', color='black')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str,
                        default='/mnt/workspace/workgroup/yuanyq/code/video_seg/EasyVLM/evaluation_results/1223_pretrain_v1/refcoco_val.json')
    parser.add_argument('--save', type=str, default='visualization/refcoco/')
    args = parser.parse_args()

    with open(args.json, 'r', encoding='utf8') as f:
        pred_datas = json.load(f)

    os.makedirs(args.save, exist_ok=True)

    for pred_data in tqdm(pred_datas):
        if 'idx' not in pred_data:
            continue

        idx = pred_data['idx']
        img_path = pred_data['image_path']
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # ---- GT: support gt_mask_rle being list / single ----
        gt_masks = annAnyToMasks(pred_data.get('gt_mask_rle', None), h, w)
        gt_mask = merge_masks(gt_masks, h, w)  # union into one GT mask

        # ---- Pred: also use unified parsing ----
        if 'mask_rle' in pred_data:
            pred_masks = annAnyToMasks(pred_data.get('mask_rle', None), h, w)
        else:
            pred_masks = annAnyToMasks(pred_data.get('pred_rles', None), h, w)

        iou = pred_data.get('iou', None)
        save_path = os.path.join(args.save, f"{idx}.jpg")
        visualize_masks(img_path, gt_mask, pred_masks, iou, pred_data.get('instruction', ''), save_path)
        print(f'save {save_path}...')

if __name__ == '__main__':
    main()