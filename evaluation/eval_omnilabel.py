import argparse              # 命令行参数解析库
import os                    # 操作系统接口，用于路径操作
import json                  # JSON文件读写
import torch                 # PyTorch深度学习框架
import numpy as np           # 数值计算库
from PIL import Image, ImageFile  # 图像处理库
from tqdm import tqdm        # 进度条显示库
import torch.nn.functional as F   # PyTorch函数式接口，用于图像插值
import torch.distributed as dist  # PyTorch分布式训练模块
import sys                   # 系统模块
from omnilabeltools import OmniLabel
# 允许加载截断/损坏的图片，避免程序因个别损坏图片而崩溃
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 将项目根目录添加到Python路径，以便导入easy_vlm模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型相关函数
from easy_vlm.models import load_pretrained_model  # 加载预训练模型（VLM + SAM）
from easy_vlm import mm_infer_segmentation         # 多模态分割推理函数


def mask_to_bbox(mask: torch.Tensor):
    # 允许输入形状为 [1,H,W] / [H,W] 等
    mask2d = mask.squeeze()
    if mask2d.ndim != 2:
        raise ValueError(f"mask must be 2D after squeeze, got shape {tuple(mask2d.shape)}")

    # 转为 bool：非零即 True
    m = mask2d != 0

    # 空 mask
    if not m.any().item():
        return None

    # 找到前景像素坐标
    ys, xs = torch.where(m)  # 1D indices
    y_min = int(ys.min().item())
    y_max = int(ys.max().item())
    x_min = int(xs.min().item())
    x_max = int(xs.max().item())

    # xywh（包含边界像素，所以 +1）
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def load_omnilabel_samples(gt_json_path, image_folder, subset=None, limit=None):
    """
    使用 omnilabeltools 直接读取每张图的 labelspace（即该图对应的 descriptions）。
    返回: List[dict] samples
    """
    print(f"Loading OmniLabel GT via omnilabeltools from {gt_json_path}...")

    ol = OmniLabel(gt_json_path)

    samples = []
    img_ids = sorted(list(ol.image_ids))

    for img_id in img_ids:
        img_info = ol.get_image_sample(img_id)  # 包含 file_name / labelspace / (可能有 width/height)

        file_name = img_info["file_name"]
        img_path = os.path.join(image_folder, file_name)

        labelspace = img_info.get("labelspace", [])

        samples.append({
            "img_id": img_info["id"],
            "img_path": img_path,
            "labelspace": labelspace,
            "width": img_info.get("width", None),
            "height": img_info.get("height", None),
        })

        if limit is not None and len(samples) >= limit: #debug
            break 

    print(f"  Loaded images: {len(samples)}")
    return samples


def main():
    """
    主函数：解析参数、加载模型、执行推理、保存结果、运行评测
    """
    
    # ==================== 1. 解析命令行参数 ====================
    parser = argparse.ArgumentParser(description='Evaluate OmniLabel Benchmark')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型checkpoint的路径')
    parser.add_argument('--gt_json', type=str, required=True,
                        help='OmniLabel GT标注JSON文件的路径')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='图片目录的基础路径')
    
    # 可选参数
    parser.add_argument('--output_file', type=str, default='omnilabel_results.json',
                        help='预测结果保存路径')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='mask置信度阈值')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制处理的图片数量（调试用）')
    parser.add_argument('--subset', type=str, default=None,
                        choices=['coco', 'object365', 'openimagesv5'],
                        help='只评测特定子集')
    parser.add_argument('--no_eval', action='store_true',
                        help='只推理不评测')
    parser.add_argument('--resume', action='store_true',
                        help='从已有结果继续，跳过已处理的图片')
    
    args = parser.parse_args()

    # ==================== 2. 初始化分布式环境 ====================
    # 检查是否在分布式环境中运行
    if 'WORLD_SIZE' in os.environ:
        # 初始化分布式进程组
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()              # 当前进程排名
        world_size = dist.get_world_size()  # 总进程数
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 本节点排名
        torch.cuda.set_device(local_rank)   # 设置使用的GPU
        device = torch.device(f"cuda:{local_rank}")
    else:
        # 单GPU模式
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== 3. 加载模型 ====================
    if rank == 0:
        print(f"Loading model from {args.model_path}...")
    
    # 加载预训练的VLM+SAM分割模型
    tokenizer, model, processor = load_pretrained_model(
        args.model_path,              # 模型路径
        None,                         # 模型名称
        attn_implementation='sdpa',   # 使用SDPA注意力（更快）
        device_map=device             # 模型放置的设备
    )
    
    # 转换为bfloat16精度（减少显存，加速推理）
    model.to(dtype=torch.bfloat16)

    # ==================== 4. 加载OmniLabel标注 ====================
    samples = load_omnilabel_samples(
        gt_json_path=args.gt_json,
        image_folder=args.image_folder,
        subset=args.subset,
        limit=args.limit
    )
    # 分布式切片：轮询分配样本
    my_samples = samples[rank::world_size]
    print(f"[Rank {rank}] Processing {len(my_samples)} / {len(samples)} images...")
    
    # ==================== 6. 断点续传 ====================
    results = []
    processed_img_ids = set()

    if args.resume and os.path.exists(args.output_file):
        if rank == 0:
            print(f"Resuming from {args.output_file}...")
        with open(args.output_file, 'r') as f:
            existing_results = json.load(f)
        results = existing_results
        processed_img_ids = {r['image_id'] for r in existing_results}
        if rank == 0:
            print(f"Loaded {len(existing_results)} existing predictions")

    # 过滤已处理的图片
    my_samples = [s for s in my_samples if s["img_id"] not in processed_img_ids]
    if rank == 0:
        print(f"Remaining images to process: {len(my_samples)}")

    iterator = tqdm(my_samples, desc=f"Rank {rank}", position=rank, leave=True)

    
    for sample in iterator:
        img_id = sample["img_id"]
        image_path = sample["img_path"]

        # 获取原图尺寸：优先用 GT 提供的 width/height（若有），否则读图获取
        img_w = sample.get("width", None)
        img_h = sample.get("height", None)
        if img_w is None or img_h is None:
            with Image.open(image_path) as img:
                img_w, img_h = img.size

        labelspace = sample["labelspace"]
        if not labelspace:
            continue

        for ls in labelspace:
            desc_id = ls["id"]
            desc_text = ls["text"]

            query_text = f"Please segment '{desc_text}' in the image."
            contents = [
                {"type": "image", "image": image_path},
                {"type": "text", "text": query_text}
            ]
            conversation = [{"role": "user", "content": contents}]

            output, masks, cls_scores = mm_infer_segmentation(
                image_path,
                processor,
                conversation,
                model,
                tokenizer
            )

            if masks is None:
                continue
                
            keep = cls_scores > args.threshold
            selected_masks = masks[keep]
            if selected_masks.numel() == 0:
                continue
            selected_scores = cls_scores[keep]

            selected_masks = F.interpolate(selected_masks.unsqueeze(0), size=(img_h, img_w), mode='bilinear', align_corners=False).squeeze(0)>0
            for selected_mask, selected_score in zip(selected_masks, selected_scores):
                bbox = mask_to_bbox(selected_mask>0)

                if bbox is None:
                    continue

                results.append({
                    "image_id": img_id,
                    "bbox": bbox,
                    "description_ids": [desc_id],
                    "scores": [float(selected_score)]
                })


    # ==================== 12. 保存当前rank结果 ====================
    temp_file = args.output_file.replace('.json', f'_rank{rank}.json')
    print(f"[Rank {rank}] Saving {len(results)} results to {temp_file}...")
    with open(temp_file, 'w') as f:
        json.dump(results, f)
    
    # 同步等待所有进程
    if world_size > 1:
        dist.barrier()
    
    # ==================== 13. 合并结果并评测（rank 0）====================
    if rank == 0:
        print("Merging results from all ranks...")
        final_results = []
        
        # 合并所有rank的结果
        for r in range(world_size):
            part_file = args.output_file.replace('.json', f'_rank{r}.json')
            if os.path.exists(part_file):
                with open(part_file, 'r') as f:
                    final_results.extend(json.load(f))
                os.remove(part_file)
        
        print(f"Total predictions: {len(final_results)}")
        
        # 保存最终结果
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(final_results, f)
        print(f"Results saved to {args.output_file}")
        
        # ==================== 14. 运行OmniLabel评测 ====================
        if not args.no_eval:
            print("\n" + "="*50)
            print("Running OmniLabel Evaluation...")
            print("="*50)
            
            # try:
            from omnilabeltools import OmniLabel, OmniLabelEval
            
            gt = OmniLabel(args.gt_json)
            dt = gt.load_res(args.output_file)
            
            ole = OmniLabelEval(gt, dt)
            ole.evaluate()
            ole.accumulate()
            ole.summarize()
            
            # except ImportError:
            #     print("\nWarning: omnilabeltools not installed.")
            #     print(f"Run: oleval --path-to-gt {args.gt_json} --path-to-res {args.output_file}")
        else:
            print(f"\n--no_eval specified. Results saved to: {args.output_file}")
    
    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
