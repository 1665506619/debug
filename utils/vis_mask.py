import json
import os
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as maskUtils


def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def safe_filename(name: str, max_len: int = 120) -> str:
    name = name.strip().replace('"', "").replace("'", "")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-_.]+", "", name)
    name = name.strip("._-") or "phrase"
    return name[:max_len]


def overlay_masks(img_arr, masks, alpha, rng):
    for mobj in masks:
        mask = annToMask(mobj).astype(bool)
        color = np.array([rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)], dtype=np.float32)
        img_arr[mask, :] = (1 - alpha) * img_arr[mask, :] + alpha * color


def draw_bboxes_xywh(pil_img, bboxes_xywh, outline=(255, 0, 0), width=4, label=None):
    """
    bboxes_xywh: list of [x, y, w, h]
    """
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for bb in bboxes_xywh:
        x, y, w, h = map(float, bb)
        x1, y1, x2, y2 = x, y, x + w, y + h

        draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)

        if label:
            text = str(label)
            if font:
                l, t, r, b = draw.textbbox((0, 0), text, font=font)
                tw, th = r - l, b - t
            else:
                tw, th = (len(text) * 6, 12)

            tx, ty = x1, max(0, y1 - th - 6)
            draw.rectangle([tx, ty, tx + tw + 6, ty + th + 6], fill=(0, 0, 0))
            draw.text((tx + 3, ty + 3), text, fill=(255, 255, 255), font=font)

    return pil_img


def visualize_masks_bboxes_per_text(data, out_dir=".", alpha=0.45, seed=42,
                                    bbox_color=(255, 0, 0), bbox_width=4, draw_label=True):

    img_path = data["image"]
    img_path = os.path.join('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data', img_path)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_img = Image.open(img_path).convert("RGB")
    base_arr = np.array(base_img).astype(np.float32)

    anns = data.get("annotation", [])
    if not anns:
        raise ValueError("No annotation found in JSON.")

    for i, ann in enumerate(anns):
        text = ann.get("text", f"ann_{i}")
        fname = safe_filename(text) + ".png"

        rng = random.Random(seed + i)
        img_arr = base_arr.copy()

        # 1) overlay masks
        overlay_masks(img_arr, ann.get("mask", []), alpha=alpha, rng=rng)

        # 2) draw bbox (xywh)
        out_img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
        bboxes_xywh = ann.get("bbox", [])
        label = text if draw_label else None
        out_img = draw_bboxes_xywh(out_img, bboxes_xywh, outline=bbox_color, width=bbox_width, label=label)

        out_path = out_dir / fname
        out_img.save(out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    for d_ in open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/humanref_45k_mask/shard_00.jsonl'):
        data = json.loads(d_)
        visualize_masks_bboxes_per_text(data, out_dir="./vis", alpha=0.45)
