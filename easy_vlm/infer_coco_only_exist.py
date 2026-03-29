import sys
sys.path.append('./')

import torch
from transformers import AutoProcessor
import json
import os
from easy_vlm.models import load_pretrained_model
from easy_vlm import mm_infer_segmentation
from PIL import Image
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils


def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann["counts"], list):
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def to_score_list(score, masks, mask_list):
    if isinstance(score, dict):
        keys = list(masks.keys())
        return [float(score[k]) for k in keys]

    if torch.is_tensor(score):
        score = score.detach().cpu().tolist()
    if isinstance(score, np.ndarray):
        score = score.tolist()

    if isinstance(score, (int, float)):
        return [float(score)] * len(mask_list)

    if isinstance(score, (list, tuple)):
        if len(score) == 1 and isinstance(score[0], (list, tuple)):
            score = score[0]
        return [float(x) for x in score]

    return [None] * len(mask_list)


def main():
    modal = "image"

    data = json.load(open('/mnt/workspace/workgroup/yuanyq/code/video_seg/datasets/new_format/coco_instances_val.json'))
    model_path = "./work_dirs/1224_pretrain_v1_10_query_soft_cls_lora"

    tokenizer, model, processor = load_pretrained_model(model_path, None, attn_implementation='sdpa')
    processor = AutoProcessor.from_pretrained(model_path)

    os.makedirs("vis_only_exist_100", exist_ok=True)

    score_thr = 0.3
    cols = 5  # 每行固定 5 张；行数按可视化数量自适应

    for idx, d in enumerate(tqdm(data)):
        try:
            image_file = os.path.join('/mnt/workspace/workgroup/yuanyq/video_data', d['image'])
            images = Image.open(image_file).convert('RGB')
        except:
            image_file = image_file.replace('train2017', 'val2017')
            images = Image.open(image_file).convert('RGB')

        for ann in d['annotation']:
            instruction = f"Please segment '{ann['text']}' the image. "
            contents = [
                {"type": "image", "image": image_file},
                {"type": "text", "text": instruction},
            ]
            conversation = [{"role": "user", "content": contents}]

            output, masks, score = mm_infer_segmentation(
                image_file,
                processor,
                conversation,
                model,
                tokenizer
            )
            print(output)
            if masks is None:
                continue

            original_image = images.convert("RGBA")
            width, height = original_image.size  # PIL: (W, H)
            alpha_value = 128

            # =============== 0) GT 可视化 ===============
            gt_mask = np.zeros((height, width), dtype=bool)
            for gt_msk in ann['mask']:
                gt_mask_ = annToMask(gt_msk, h=height, w=width)
                gt_mask = gt_mask | gt_mask_
            gt_mask_u8 = gt_mask.astype(np.uint8) * 255
            gt_pil_mask = Image.fromarray(gt_mask_u8, mode="L")
            gt_alpha = gt_pil_mask.point(lambda p: int(p > 0) * alpha_value)

            gt_overlay = Image.new("RGBA", (width, height), (0, 255, 0, 0))  # GT: 绿色
            gt_overlay.putalpha(gt_alpha)
            gt_combined = Image.alpha_composite(original_image, gt_overlay).convert("RGB")

            # =============== 1) 预测 masks/score 处理（总共 100 个，仅可视化 score>0.3） ===============
            # masks: 期望 [B, 100, h, w] 或类似；score: 期望长度 100（或 [1,100]）
            if torch.is_tensor(score):
                score_list = score.detach().float().cpu().view(-1).tolist()
            elif isinstance(score, np.ndarray):
                score_list = score.astype(np.float32).reshape(-1).tolist()
            else:
                # 兼容原先 score[0][i] 的形式
                try:
                    score_list = torch.tensor(score).float().view(-1).tolist()
                except Exception:
                    score_list = []

            num_masks = masks.shape[1]
            if len(score_list) >= num_masks:
                score_list = score_list[:num_masks]
            else:
                score_list = score_list + [None] * (num_masks - len(score_list))

            keep_indices = [i for i, s in enumerate(score_list) if (s is not None and float(s) > score_thr)]
            if len(keep_indices) == 0:
                continue

            viz_imgs = [gt_combined]
            viz_titles = ["GT ann['mask']"]

            for i in keep_indices:
                mask = masks[:, i:i+1]  # [B,1,h,w] 或 [B,1,H,W]
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)

                pred_masks = F.interpolate(
                    mask.float(),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                )
                pred_masks = (pred_masks > 0)

                pred_mask_np = pred_masks[0, 0].detach().cpu().numpy().astype(np.uint8) * 255
                pil_mask = Image.fromarray(pred_mask_np, mode="L")
                alpha_mask = pil_mask.point(lambda p: int(p > 0) * alpha_value)

                overlay = Image.new("RGBA", (width, height), (255, 0, 0, 0))  # Pred: 红色
                overlay.putalpha(alpha_mask)

                combined = Image.alpha_composite(original_image, overlay).convert("RGB")
                viz_imgs.append(combined)

                s = float(score_list[i]) if score_list[i] is not None else None
                viz_titles.append(f"pred idx={i}, score={s:.4f}" if s is not None else f"pred idx={i}, score=None")

            # =============== 2) 按可视化数量排版 ===============
            n = len(viz_imgs)  # GT + 过滤后的 preds
            rows = int(np.ceil(n / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            axes = np.array(axes).reshape(-1)

            for k in range(rows * cols):
                ax = axes[k]
                ax.axis("off")
                if k < n:
                    ax.imshow(viz_imgs[k])
                    ax.set_title(viz_titles[k], fontsize=12)

            plt.tight_layout()
            save_path = os.path.join(
                "vis_only_exist_100",
                f"{d['image'].replace('.jpg', '').replace('coco/train2017/', '')}_{ann['text']}.png"
            )
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
