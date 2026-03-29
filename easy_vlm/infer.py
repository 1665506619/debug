import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import sys
sys.path.append('./')
from easy_vlm.models import load_pretrained_model
from easy_vlm import mm_infer_segmentation
from PIL import Image
import torch.nn.functional as F
import numpy as np

def infer_and_vis(query, image_path, processor, model, tokenizer):
    contents = []
    contents.append({"type": "image", "image": image_path})
    contents.append({"type": "text", "text": query})

    conversation = [{"role": "user", "content": contents}]

    output, masks, cls_score = mm_infer_segmentation(
        image_path,
        processor,
        conversation,
        model,
        tokenizer
    )
    print(output)
    import pdb 
    pdb.set_trace()
    print(cls_score)
    if masks is not None:
        original_image = Image.open(image_path).convert("RGBA")
        width, height = original_image.size   # 注意 PIL 是 (width, height)

        alpha_value = 128  # 0~255，数值越大越不透明

        for i in range(masks.shape[1]):
            mask = masks[:,i:i+1]

            # 插值到和原图一样大（size 是 (H, W)）
            pred_masks = F.interpolate(
                mask.float(), 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )

            # 二值化
            pred_masks = (pred_masks > 0)

            # 转成 0/255 的 uint8
            pred_mask_np = pred_masks[0, 0].detach().cpu().numpy().astype(np.uint8) * 255

            # 转成 PIL 灰度图，再转 RGBA
            pil_mask = Image.fromarray(pred_mask_np, mode="L")

            # 创建一个有颜色的遮罩层（例如红色），并使用灰度 mask 作为 alpha
            # 如果你只想整个 mask 区域统一透明度，也可以直接用 L 作为 alpha
            color_overlay = Image.new("RGBA", (width, height), (255, 0, 0, 0))  # 红色，可改成其他颜色
            # 把灰度 mask 缩放透明度到 alpha_value
            alpha_mask = pil_mask.point(lambda p: int(p > 0) * alpha_value)

            # 把带透明度的颜色层叠加到原图上
            overlay = Image.new("RGBA", (width, height), (255, 0, 0, 0))
            overlay.putalpha(alpha_mask)

            combined = Image.alpha_composite(original_image, overlay)
            combined.save(f'vis/{query}_{i}.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='work_dirs/0107_pretrain_v6_multi_obj_lora/checkpoint-8443')
    parser.add_argument("--query", type=str, default="Please segment 'train and tree' in the image.") #person, tree, car, road, tv, cup, table and phone
    parser.add_argument("--image-path", type=str, default='/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/coco/train2017/000000571746.jpg')
    args = parser.parse_args()

    tokenizer, model, processor = load_pretrained_model(args.model_path, None, attn_implementation='sdpa')

    model.to(torch.bfloat16)
    query = args.query
    image_path = args.image_path

    infer_and_vis(query, image_path, processor, model, tokenizer)
    

    

if __name__ == "__main__":
    main()
