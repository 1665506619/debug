import sys
sys.path.append('./')

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import json
import os
from easy_vlm.models import load_pretrained_model
from easy_vlm import mm_infer_segmentation
from PIL import Image
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from pycocotools import mask as maskUtils  


def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
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
    # disable_torch_init()

    modal = "image"

    data = json.load(open('/mnt/workspace/workgroup/yuanyq/code/video_seg/datasets/new_format/coco_instances_val.json'))
    model_path = "./work_dirs/1225_pretrain_v2_10_query_soft_cls_lora"
   
    tokenizer, model, processor = load_pretrained_model(model_path, None, attn_implementation='sdpa')

    processor = AutoProcessor.from_pretrained(
        model_path,
    )

    os.makedirs("vis2", exist_ok=True)

    for idx, d in enumerate(tqdm(data)):
        try:
            image_file = os.path.join('/mnt/workspace/workgroup/yuanyq/video_data', d['image'])
            images = Image.open(image_file).convert('RGB')
        except:
            image_file = image_file.replace('train2017', 'val2017')
            images = Image.open(image_file).convert('RGB')

        for ann in d['annotation']:
            instruction = f"Please segment '{ann['text']}' the image. "
            contents = []
            contents.append({"type": "image", "image": image_file})
            contents.append({"type": "text", "text": instruction})

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

            # =============== 0) 先做一张 GT ann['mask'] 的可视化 ===============
            gt_mask = np.zeros((height, width), dtype=bool)
            for gt_msk in ann['mask']:
                gt_mask_ = annToMask(gt_msk, h=height, w=width)  # ndarray, HxW (0/1)
                gt_mask = gt_mask | gt_mask_
            gt_mask_u8 = gt_mask.astype(np.uint8) * 255
            gt_pil_mask = Image.fromarray(gt_mask_u8, mode="L")
            gt_alpha = gt_pil_mask.point(lambda p: int(p > 0) * alpha_value)

            gt_overlay = Image.new("RGBA", (width, height), (0, 255, 0, 0))  # GT 用绿色
            gt_overlay.putalpha(gt_alpha)
            gt_combined = Image.alpha_composite(original_image, gt_overlay).convert("RGB")

            # 1) 生成预测叠加图
            viz_imgs = []
            for i in range(masks.shape[1]):
                mask = masks[:,i:i+1]
                if mask.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
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

                overlay = Image.new("RGBA", (width, height), (255, 0, 0, 0))  # Pred 用红色
                overlay.putalpha(alpha_mask)

                combined = Image.alpha_composite(original_image, overlay).convert("RGB")
                viz_imgs.append(combined)

            # =============== 2) 把 GT 图插到最前面，并对齐标题/score ===============
            viz_imgs = [gt_combined] + viz_imgs

            # 3) matplotlib 拼图
            n = len(viz_imgs)
            cols = 5
            rows = int(np.ceil(n / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            axes = np.array(axes).reshape(-1)

            for i in range(rows * cols):
                ax = axes[i]
                ax.axis("off")
                if i < n:
                    ax.imshow(viz_imgs[i])
                    if i == 0:
                        ax.set_title("GT ann['mask']", fontsize=12)
                    else:
                        s = float(score[0][i-1])
                        title = f"pred idx={i-1}, score={s:.4f}" if s is not None else f"pred idx={i-1}, score=None"
                        ax.set_title(title, fontsize=12)

            plt.tight_layout()
            save_path = os.path.join(
                "vis2",
                f"{d['image'].replace('.jpg', '').replace('coco/train2017/', '')}_{ann['text']}.png"
            )
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)  # 建议加，避免内存涨


if __name__ == "__main__":
    main()
