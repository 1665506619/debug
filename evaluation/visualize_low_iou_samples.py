#!/usr/bin/env python3
"""
可视化IoU低但有预测的样本
专门用于分析为什么IoU这么低
"""

import json
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
from tqdm import tqdm


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def decode_rle(rle):
    """解码RLE格式的mask"""
    if rle is None:
        return None
    try:
        return maskUtils.decode(rle)
    except:
        return None


def overlay_mask(image, mask, color, alpha=0.5):
    """在图像上叠加mask"""
    if mask is None or np.sum(mask) == 0:
        return image
    
    overlay = image.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
    return overlay.astype(np.uint8)


def visualize_sample(sample, questions, meta_exp, annotation_root, output_dir, max_frames=8, avg_iou=0.0):
    """可视化单个样本"""
    idx = sample['idx']
    
    if idx >= len(questions):
        return False
    
    question_item = questions[idx]
    video_id = question_item['video_id']
    exp_id = question_item['exp_id']
    obj_id = question_item['obj_id']
    expression = question_item.get('conversations', [{}])[0].get('value', 'Unknown')
    
    # 提取expression文本
    if 'Can you segment' in expression:
        expression = expression.replace('Can you segment ', '').replace(' in the video?', '')
    
    if video_id not in meta_exp['videos']:
        return False
    
    meta_frames = meta_exp['videos'][video_id]['frames']
    all_frame_names = question_item.get('frame_names', meta_frames)
    mask_rles = sample.get('mask_rle', [])
    
    # 加载GT
    gt_mask_file = os.path.join(annotation_root, 'annotations', video_id, f'{obj_id}.json')
    try:
        with open(gt_mask_file, 'r') as f:
            gt_mask_dict = json.load(f)
    except:
        return False
    
    # 图像根目录
    image_root = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/EgoMask/dataset/egomask/JPEGImages/refego'
    
    # 只选择有预测的帧
    frames_with_pred = []
    for i, frame_name in enumerate(all_frame_names):
        if i >= len(mask_rles):
            break
        
        pred_rle = mask_rles[i]
        if pred_rle is None:
            continue
        
        pred_mask = decode_rle(pred_rle)
        if pred_mask is None or np.sum(pred_mask) == 0:
            continue
        
        # 计算IoU
        gt_mask = None
        if frame_name in gt_mask_dict and gt_mask_dict[frame_name] is not None:
            gt_mask = decode_rle(gt_mask_dict[frame_name])
        
        iou = 0.0
        if gt_mask is not None and np.sum(gt_mask) > 0:
            intersection = np.sum((gt_mask > 0) & (pred_mask > 0))
            union = np.sum((gt_mask > 0) | (pred_mask > 0))
            if union > 0:
                iou = intersection / union
        
        frames_with_pred.append((i, frame_name, iou))
    
    if len(frames_with_pred) == 0:
        return False
    
    # 选择要可视化的帧（均匀采样）
    if len(frames_with_pred) > max_frames:
        indices = np.linspace(0, len(frames_with_pred) - 1, max_frames, dtype=int)
        selected_frames = [frames_with_pred[i] for i in indices]
    else:
        selected_frames = frames_with_pred
    
    # 创建图像（减小尺寸以节省内存）
    fig, axes = plt.subplots(3, len(selected_frames), figsize=(3 * len(selected_frames), 9))
    if len(selected_frames) == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, (frame_idx, frame_name, iou) in enumerate(selected_frames):
        # 加载图像
        image_path = os.path.join(image_root, video_id, f'{frame_name}.jpg')
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.ones((360, 640, 3), dtype=np.uint8) * 255
        
        # 加载GT mask
        gt_mask = None
        if frame_name in gt_mask_dict and gt_mask_dict[frame_name] is not None:
            gt_mask = decode_rle(gt_mask_dict[frame_name])
        
        # 加载预测mask
        pred_mask = decode_rle(mask_rles[frame_idx])
        
        # 第一行：原始图像
        axes[0, col_idx].imshow(image)
        axes[0, col_idx].set_title(f'Frame {frame_idx}: {frame_name}', fontsize=10)
        axes[0, col_idx].axis('off')
        
        # 第二行：GT mask（绿色）
        if gt_mask is not None and np.sum(gt_mask) > 0:
            gt_overlay = overlay_mask(image, gt_mask, [0, 255, 0], alpha=0.5)
            axes[1, col_idx].imshow(gt_overlay)
            gt_area = np.sum(gt_mask > 0)
            axes[1, col_idx].set_title(f'GT (Green)\nArea: {gt_area}px', fontsize=10, color='green')
        else:
            axes[1, col_idx].imshow(image)
            axes[1, col_idx].set_title(f'GT: None', fontsize=10, color='gray')
        axes[1, col_idx].axis('off')
        
        # 第三行：预测mask（红色）
        if pred_mask is not None and np.sum(pred_mask) > 0:
            pred_overlay = overlay_mask(image, pred_mask, [255, 0, 0], alpha=0.5)
            axes[2, col_idx].imshow(pred_overlay)
            pred_area = np.sum(pred_mask > 0)
            
            if gt_mask is None or np.sum(gt_mask) == 0:
                axes[2, col_idx].set_title(f'Pred (Red)\nArea: {pred_area}px\n(False Positive)', 
                                          fontsize=10, color='orange')
            else:
                color = 'green' if iou > 0.5 else 'red'
                axes[2, col_idx].set_title(f'Pred (Red)\nArea: {pred_area}px\nIoU: {iou:.3f}', 
                                          fontsize=10, color=color)
        else:
            axes[2, col_idx].imshow(image)
            axes[2, col_idx].set_title(f'Pred: None', fontsize=10, color='gray')
        axes[2, col_idx].axis('off')
    
    # 设置总标题
    j_score = sample.get('j', 0)
    f_score = sample.get('f', 0)
    fig.suptitle(f'Video: {video_id} | Exp {exp_id} (obj {obj_id})\n{expression}\nJ={j_score:.3f}, F={f_score:.3f}, Avg IoU={avg_iou:.3f}', 
                 fontsize=12, y=0.98)
    
    plt.tight_layout()
    
    # 保存（降低DPI以节省内存）
    output_path = os.path.join(output_dir, f'{video_id}_exp{exp_id}_obj{obj_id}_iou{avg_iou:.3f}.png')
    plt.savefig(output_path, dpi=80, bbox_inches='tight')
    plt.close('all')  # 确保释放内存
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--question_file', type=str, required=True)
    parser.add_argument('--meta_file', type=str, required=True)
    parser.add_argument('--annotation_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--max_frames', type=int, default=4)
    parser.add_argument('--min_iou', type=float, default=0.0, help='最小IoU')
    parser.add_argument('--max_iou', type=float, default=0.3, help='最大IoU')
    args = parser.parse_args()
    
    # 加载数据
    print(f'加载数据...')
    predictions = load_json(args.pred_file)
    if predictions and 'j' in predictions[0] and 'idx' not in predictions[0]:
        predictions = predictions[1:]
    
    questions = load_json(args.question_file)
    meta_exp = load_json(args.meta_file)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 筛选符合条件的样本
    print(f'\n筛选 {args.min_iou} < IoU < {args.max_iou} 且有预测的样本...')
    valid_samples = []
    
    for sample in tqdm(predictions):
        idx = sample['idx']
        if idx >= len(questions):
            continue
        
        question_item = questions[idx]
        video_id = question_item['video_id']
        exp_id = question_item['exp_id']
        obj_id = question_item['obj_id']
        
        if video_id not in meta_exp['videos']:
            continue
        
        mask_rles = sample.get('mask_rle', [])
        
        # 加载GT
        gt_mask_file = os.path.join(args.annotation_root, 'annotations', video_id, f'{obj_id}.json')
        try:
            with open(gt_mask_file, 'r') as f:
                gt_mask_dict = json.load(f)
        except:
            continue
        
        meta_frames = meta_exp['videos'][video_id]['frames']
        all_frame_names = question_item.get('frame_names', meta_frames)
        
        # 计算平均IoU（只考虑有预测的帧）
        ious = []
        for i, frame_name in enumerate(all_frame_names):
            if i >= len(mask_rles):
                break
            
            pred_rle = mask_rles[i]
            if pred_rle is None:
                continue
            
            pred_mask = decode_rle(pred_rle)
            if pred_mask is None or np.sum(pred_mask) == 0:
                continue
            
            # 计算IoU
            gt_mask = None
            if frame_name in gt_mask_dict and gt_mask_dict[frame_name] is not None:
                gt_mask = decode_rle(gt_mask_dict[frame_name])
            
            if gt_mask is not None and np.sum(gt_mask) > 0:
                intersection = np.sum((gt_mask > 0) & (pred_mask > 0))
                union = np.sum((gt_mask > 0) | (pred_mask > 0))
                if union > 0:
                    iou = intersection / union
                    ious.append(iou)
        
        if len(ious) > 0:
            avg_iou = np.mean(ious)
            if args.min_iou <= avg_iou <= args.max_iou:
                valid_samples.append((sample, avg_iou))
    
    print(f'找到 {len(valid_samples)} 个符合条件的样本')
    
    # 按IoU从低到高排序
    valid_samples.sort(key=lambda x: x[1])
    
    # 可视化
    print(f'\n开始可视化前 {args.num_samples} 个样本...\n')
    count = 0
    for sample, avg_iou in tqdm(valid_samples[:args.num_samples]):
        success = visualize_sample(sample, questions, meta_exp, args.annotation_root, 
                                   args.output_dir, args.max_frames, avg_iou)
        if success:
            count += 1
    
    print(f'\n完成！成功可视化 {count} 个样本，保存在 {args.output_dir}')


if __name__ == '__main__':
    main()

