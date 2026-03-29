#!/usr/bin/env python3
"""
可视化预测结果
对比 GT mask 和预测 mask
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import pycocotools.mask as maskUtils
from pathlib import Path


def load_image(image_path):
    """加载图像"""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        print(f"警告: 无法加载图像 {image_path}: {e}")
        return None


def decode_rle(rle_dict):
    """解码 RLE mask"""
    if rle_dict is None:
        return None
    try:
        mask = maskUtils.decode(rle_dict)
        return mask
    except Exception as e:
        print(f"警告: 解码 RLE 失败: {e}")
        return None


def visualize_sample(pred_item, question_item, annotation_root, image_root, output_dir, dataset_type):
    """
    可视化单个样本
    
    Args:
        pred_item: 预测结果
        question_item: question 信息
        annotation_root: GT annotations 根目录
        image_root: 图像根目录
        output_dir: 输出目录
        dataset_type: 数据集类型
    """
    video_id = question_item['video_id']
    exp_id = question_item['exp_id']
    obj_id = question_item['obj_id']
    category = question_item.get('category', 'unknown')
    frame_names = question_item.get('frame_names', [])
    
    print(f"\n可视化样本: {video_id} - {category}")
    print(f"  Expression ID: {exp_id}")
    print(f"  Object ID: {obj_id}")
    print(f"  帧数: {len(frame_names)}")
    
    # 加载 GT masks
    if dataset_type == 'medium' and '--' in video_id:
        annot_vid_name = video_id.split("--")[0]
    else:
        annot_vid_name = video_id
    
    gt_mask_file = os.path.join(annotation_root, 'annotations', annot_vid_name, f'{obj_id}.json')
    
    if not os.path.exists(gt_mask_file):
        print(f"  警告: GT mask 文件不存在: {gt_mask_file}")
        return
    
    with open(gt_mask_file, 'r') as f:
        gt_mask_dict = json.load(f)
    
    # 获取预测 masks
    mask_rles = pred_item.get('mask_rle', [])
    
    # 创建 frame_name 到 mask 的映射
    frame_to_pred_mask = {}
    for i, fn in enumerate(frame_names):
        if i < len(mask_rles) and mask_rles[i] is not None:
            frame_to_pred_mask[fn] = mask_rles[i]
    
    # 确定要可视化的帧（选择有 GT 的帧）
    frames_to_vis = []
    for fname in frame_names:
        if fname in gt_mask_dict and gt_mask_dict[fname] is not None:
            frames_to_vis.append(fname)
            if len(frames_to_vis) >= 6:  # 最多显示 6 帧
                break
    
    if not frames_to_vis:
        print(f"  警告: 没有可视化的帧")
        return
    
    # 确定图像路径
    if dataset_type == 'short':
        image_dir = os.path.join(image_root, 'JPEGImages/refego', video_id)
    else:
        image_dir = os.path.join(image_root, 'JPEGImages/egotracks', annot_vid_name)
    
    # 创建可视化
    n_frames = len(frames_to_vis)
    fig, axes = plt.subplots(n_frames, 4, figsize=(16, 4 * n_frames))
    
    if n_frames == 1:
        axes = axes.reshape(1, -1)
    
    for idx, fname in enumerate(frames_to_vis):
        # 加载图像
        image_path = os.path.join(image_dir, f'{fname}.jpg')
        img = load_image(image_path)
        
        if img is None:
            # 尝试其他扩展名
            for ext in ['.jpeg', '.JPG', '.JPEG', '.png']:
                image_path = os.path.join(image_dir, f'{fname}{ext}')
                img = load_image(image_path)
                if img is not None:
                    break
        
        if img is None:
            print(f"  警告: 无法加载图像 {fname}")
            continue
        
        # 加载 GT mask
        gt_mask = decode_rle(gt_mask_dict.get(fname))
        
        # 加载预测 mask
        pred_mask = decode_rle(frame_to_pred_mask.get(fname))
        
        # 显示原图
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'Frame: {fname}', fontsize=10)
        axes[idx, 0].axis('off')
        
        # 显示 GT mask
        if gt_mask is not None:
            axes[idx, 1].imshow(img)
            axes[idx, 1].imshow(gt_mask, alpha=0.5, cmap='jet')
            axes[idx, 1].set_title(f'GT Mask (pixels: {gt_mask.sum()})', fontsize=10)
            axes[idx, 1].axis('off')
        else:
            axes[idx, 1].text(0.5, 0.5, 'No GT', ha='center', va='center', fontsize=12)
            axes[idx, 1].axis('off')
        
        # 显示预测 mask
        if pred_mask is not None:
            axes[idx, 2].imshow(img)
            axes[idx, 2].imshow(pred_mask, alpha=0.5, cmap='jet')
            axes[idx, 2].set_title(f'Pred Mask (pixels: {pred_mask.sum()})', fontsize=10)
            axes[idx, 2].axis('off')
        else:
            axes[idx, 2].text(0.5, 0.5, 'No Pred', ha='center', va='center', fontsize=12)
            axes[idx, 2].axis('off')
        
        # 显示对比（红色=GT only, 绿色=Pred only, 黄色=Both）
        if gt_mask is not None and pred_mask is not None:
            h, w = gt_mask.shape
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            
            # GT only: 红色
            overlay[gt_mask > 0] = [255, 0, 0]
            # Pred only: 绿色
            overlay[pred_mask > 0] = [0, 255, 0]
            # Both: 黄色
            overlap = np.logical_and(gt_mask > 0, pred_mask > 0)
            overlay[overlap] = [255, 255, 0]
            
            axes[idx, 3].imshow(img)
            axes[idx, 3].imshow(overlay, alpha=0.5)
            
            # 计算 IoU
            intersection = overlap.sum()
            union = np.logical_or(gt_mask > 0, pred_mask > 0).sum()
            iou = intersection / union if union > 0 else 0.0
            
            axes[idx, 3].set_title(f'Overlay (IoU: {iou:.3f})', fontsize=10)
            axes[idx, 3].axis('off')
            
            # 添加图例
            if idx == 0:
                red_patch = mpatches.Patch(color='red', label='GT only')
                green_patch = mpatches.Patch(color='green', label='Pred only')
                yellow_patch = mpatches.Patch(color='yellow', label='Both')
                axes[idx, 3].legend(handles=[red_patch, green_patch, yellow_patch], 
                                   loc='upper right', fontsize=8)
        else:
            axes[idx, 3].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            axes[idx, 3].axis('off')
    
    # 设置总标题
    fig.suptitle(f'Video: {video_id}\nCategory: {category}\nExpression ID: {exp_id}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{video_id}_{exp_id}_{obj_id}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 保存到: {output_path}")


def main(args):
    # 加载预测结果
    print(f"正在加载预测结果: {args.pred_result}")
    with open(args.pred_result, 'r') as f:
        predictions = json.load(f)
    
    # 跳过第一个元素（如果是 metrics summary）
    if predictions and 'j' in predictions[0] and 'idx' not in predictions[0]:
        print("检测到 metrics summary，跳过第一个元素")
        predictions = predictions[1:]
    
    print(f"共有 {len(predictions)} 条预测结果")
    
    # 加载 question 文件
    question_file = f'/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_{args.dataset_type}.json'
    print(f"正在加载 question 文件: {question_file}")
    with open(question_file, 'r') as f:
        questions = json.load(f)
    
    # 路径配置
    annotation_root = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/egomask'
    image_root = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/EgoMask/dataset/egomask'
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 可视化样本
    num_to_vis = min(args.num_samples, len(predictions))
    print(f"\n开始可视化 {num_to_vis} 个样本...\n")
    
    # 选择要可视化的样本索引
    if args.sample_indices:
        indices = [int(i) for i in args.sample_indices.split(',')]
    else:
        # 均匀采样
        step = len(predictions) // num_to_vis if num_to_vis > 0 else 1
        indices = list(range(0, len(predictions), step))[:num_to_vis]
    
    for i, idx in enumerate(indices):
        if idx >= len(predictions):
            print(f"警告: 索引 {idx} 超出范围，跳过")
            continue
        
        pred = predictions[idx]
        question_idx = pred['idx']
        
        if question_idx >= len(questions):
            print(f"警告: question 索引 {question_idx} 超出范围，跳过")
            continue
        
        question = questions[question_idx]
        
        print(f"\n[{i+1}/{num_to_vis}] 处理样本 {idx} (question_idx: {question_idx})")
        
        try:
            visualize_sample(pred, question, annotation_root, image_root, 
                           args.output_dir, args.dataset_type)
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ 可视化完成！")
    print(f"输出目录: {args.output_dir}")
    print(f"\n查看结果:")
    print(f"  ls -lh {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 EgoMask 预测结果")
    parser.add_argument("--pred_result", type=str, required=True,
                        help="预测结果 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--dataset_type", type=str, default="short",
                        choices=["long", "medium", "short", "full"],
                        help="数据集类型")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="要可视化的样本数量")
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="指定要可视化的样本索引，用逗号分隔，例如: 0,5,10,15")
    args = parser.parse_args()
    
    main(args)

