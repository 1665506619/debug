#!/usr/bin/env python3
"""
可视化 EgoMask 评估结果：对比 Ground Truth 和模型预测
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random
from pycocotools import mask as maskUtils


def visualize_sample(video_id, exp_id, obj_id, 
                     image_dir, gt_mask_file, pred_mask_file, 
                     frames, output_dir, max_frames=5):
    """
    可视化单个样本的 GT 和预测结果
    
    Args:
        video_id: 视频ID
        exp_id: 表达式ID
        obj_id: 对象ID
        image_dir: 图像目录
        gt_mask_file: GT mask JSON文件
        pred_mask_file: 预测 mask JSON文件
        frames: 帧列表
        output_dir: 输出目录
        max_frames: 最多可视化多少帧
    """
    # 读取 GT masks
    if not os.path.exists(gt_mask_file):
        print(f"GT mask 文件不存在: {gt_mask_file}")
        return
    
    with open(gt_mask_file, 'r') as f:
        gt_masks = json.load(f)
    
    # 读取预测 masks
    if not os.path.exists(pred_mask_file):
        print(f"预测 mask 文件不存在: {pred_mask_file}")
        return
    
    with open(pred_mask_file, 'r') as f:
        pred_masks = json.load(f)
    
    # 找出有 GT mask 的帧
    gt_frames = [f for f in frames if f in gt_masks and gt_masks[f] is not None]
    
    if not gt_frames:
        print(f"没有 GT mask 帧")
        return
    
    # 随机选择几帧
    selected_frames = random.sample(gt_frames, min(max_frames, len(gt_frames)))
    
    # 创建输出目录
    sample_output_dir = os.path.join(output_dir, f"{video_id}--{exp_id}")
    os.makedirs(sample_output_dir, exist_ok=True)
    
    for frame_name in selected_frames:
        # 读取原始图像（添加 .jpg 扩展名）
        img_path = os.path.join(image_dir, f"{frame_name}.jpg")
        if not os.path.exists(img_path):
            print(f"图像不存在: {img_path}")
            continue
        
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # 解码 GT mask
        gt_rle = gt_masks.get(frame_name)
        gt_mask = maskUtils.decode(gt_rle) if gt_rle else np.zeros(img_np.shape[:2], dtype=np.uint8)
        
        # 解码预测 mask
        pred_rle = pred_masks.get(frame_name)
        pred_mask = maskUtils.decode(pred_rle) if pred_rle else np.zeros(img_np.shape[:2], dtype=np.uint8)
        
        # 计算 IoU
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = intersection / union if union > 0 else 0.0
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 第一行：原图 + GT + 预测
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # GT mask 叠加
        gt_overlay = img_np.copy()
        gt_overlay[gt_mask > 0] = gt_overlay[gt_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        axes[0, 1].imshow(gt_overlay)
        axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold', color='green')
        axes[0, 1].axis('off')
        
        # 预测 mask 叠加
        pred_overlay = img_np.copy()
        pred_overlay[pred_mask > 0] = pred_overlay[pred_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[0, 2].imshow(pred_overlay)
        axes[0, 2].set_title('Prediction', fontsize=14, fontweight='bold', color='red')
        axes[0, 2].axis('off')
        
        # 第二行：GT mask only + Pred mask only + Overlay对比
        axes[1, 0].imshow(gt_mask, cmap='gray')
        axes[1, 0].set_title('GT Mask Only', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred_mask, cmap='gray')
        axes[1, 1].set_title('Pred Mask Only', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # GT (绿) + Pred (红) + Overlap (黄)
        comparison = np.zeros((*img_np.shape[:2], 3), dtype=np.uint8)
        comparison[gt_mask > 0] = [0, 255, 0]  # GT: 绿色
        comparison[pred_mask > 0] = [255, 0, 0]  # Pred: 红色
        comparison[np.logical_and(gt_mask > 0, pred_mask > 0)] = [255, 255, 0]  # Overlap: 黄色
        axes[1, 2].imshow(comparison)
        axes[1, 2].set_title(f'Comparison (IoU: {iou:.2%})', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        # 添加总标题
        fig.suptitle(f'Video: {video_id}\nExpression ID: {exp_id} | Object ID: {obj_id} | Frame: {frame_name}', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(sample_output_dir, f'{frame_name}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存: {output_path} (IoU: {iou:.2%})")


def main(args):
    # 读取 question 文件
    question_file = f'/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_{args.dataset_type}.json'
    print(f"读取 question 文件: {question_file}")
    
    with open(question_file, 'r') as f:
        questions = json.load(f)
    
    # 读取 meta_expressions
    annotation_root = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/egomask'
    if args.dataset_type == 'full':
        meta_file = os.path.join(annotation_root, 'meta_expressions.json')
    else:
        meta_file = os.path.join(annotation_root, f'subset/{args.dataset_type}/meta_expressions.json')
    
    print(f"读取 meta_expressions: {meta_file}")
    with open(meta_file, 'r') as f:
        meta_exp = json.load(f)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 随机选择样本
    if args.num_samples > 0:
        selected_indices = random.sample(range(len(questions)), min(args.num_samples, len(questions)))
    else:
        selected_indices = range(len(questions))
    
    print(f"\n开始可视化 {len(selected_indices)} 个样本...")
    
    for idx in tqdm(selected_indices, desc="可视化进度"):
        q = questions[idx]
        video_id = q['video_id']
        exp_id = q['exp_id']
        obj_id = q['obj_id']
        
        # 【修复】使用 question 中的 frame_names（所有连续帧）
        frames = q.get('frame_names', None)
        
        if frames is None:
            # 如果没有 frame_names，从 meta 中获取（兼容旧数据）
            if video_id not in meta_exp['videos']:
                print(f"\n警告: video_id {video_id} 不在 meta_expressions 中")
                continue
            video_info = meta_exp['videos'][video_id]
            frames = video_info['frames']
        
        # GT mask 文件路径
        gt_mask_dir = os.path.join(annotation_root, 'annotations')
        gt_mask_file = os.path.join(gt_mask_dir, f'{video_id}/{obj_id}.json')
        
        # 预测 mask 文件路径
        pred_dir = os.path.join(args.pred_path, args.dataset_type)
        pred_mask_file = os.path.join(pred_dir, f'{video_id}/{exp_id}/{exp_id}-{obj_id}.json')
        
        # 图像目录
        image_root = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/EgoMask/dataset/egomask/JPEGImages'
        
        # 确定图像目录（根据 subset）
        if args.dataset_type == 'medium':
            raw_clip_name = video_id.split('--')[0]
            image_dir = os.path.join(image_root, 'egotracks', raw_clip_name)
        else:
            image_dir = os.path.join(image_root, 'refego', video_id)
        
        # 可视化
        try:
            visualize_sample(
                video_id, exp_id, obj_id,
                image_dir, gt_mask_file, pred_mask_file,
                frames, args.output_dir,
                max_frames=args.frames_per_video
            )
        except Exception as e:
            print(f"\n错误处理 {video_id}--{exp_id}: {e}")
            continue
    
    print(f"\n✅ 可视化完成！结果保存到: {args.output_dir}")
    print(f"\n查看结果:")
    print(f"  ls {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 EgoMask 评估结果")
    parser.add_argument("--dataset_type", type=str, default="short",
                        choices=["long", "medium", "short", "full"],
                        help="数据集类型")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="预测结果路径")
    parser.add_argument("--output_dir", type=str, default="egomask_visualization",
                        help="可视化输出目录")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="可视化样本数量（0 = 全部）")
    parser.add_argument("--frames_per_video", type=int, default=3,
                        help="每个视频可视化多少帧")
    
    args = parser.parse_args()
    
    random.seed(42)  # 固定随机种子，便于复现
    main(args)

