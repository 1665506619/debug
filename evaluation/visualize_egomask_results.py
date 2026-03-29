"""
可视化EgoMask评估结果
将推理结果、GT mask和expression一起可视化
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pycocotools.mask as maskUtils


def rle2mask(rle):
    """将RLE格式转换为mask"""
    if rle is None:
        return None
    return maskUtils.decode(rle)


def overlay_mask(image, mask, color, alpha=0.5):
    """在图像上叠加mask"""
    if mask is None or mask.sum() == 0:
        return image
    
    # 创建彩色mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # 叠加
    overlayed = image.copy()
    overlayed = (1 - alpha) * overlayed + alpha * colored_mask
    return overlayed.astype(np.uint8)


def visualize_sample(video_id, exp_id, obj_id, 
                     frames_dir, gt_mask_path, pred_mask_path,
                     expression, output_dir, max_frames=8):
    """
    可视化单个样本
    
    Args:
        video_id: 视频ID
        exp_id: Expression ID
        obj_id: 物体ID
        frames_dir: 视频帧目录
        gt_mask_path: GT mask JSON文件路径
        pred_mask_path: 预测mask JSON文件路径
        expression: 文本描述
        output_dir: 输出目录
        max_frames: 最多可视化多少帧
    """
    # 加载GT masks
    if not os.path.exists(gt_mask_path):
        print(f"Warning: GT mask not found at {gt_mask_path}")
        return
    
    with open(gt_mask_path) as f:
        gt_masks_rle = json.load(f)
    
    # 加载预测masks
    pred_masks_rle = {}
    if os.path.exists(pred_mask_path):
        with open(pred_mask_path) as f:
            pred_masks_rle = json.load(f)
    else:
        print(f"Warning: Pred mask not found at {pred_mask_path}")
    
    # 获取所有帧
    frame_names = sorted(gt_masks_rle.keys())
    
    # 限制帧数
    if len(frame_names) > max_frames:
        # 均匀采样
        indices = np.linspace(0, len(frame_names)-1, max_frames, dtype=int)
        frame_names = [frame_names[i] for i in indices]
    
    # 创建图像网格
    n_frames = len(frame_names)
    fig, axes = plt.subplots(3, n_frames, figsize=(3*n_frames, 9))
    
    if n_frames == 1:
        axes = axes.reshape(3, 1)
    
    for idx, frame_name in enumerate(frame_names):
        # 加载原始图像
        # frame_name可能没有扩展名，尝试添加.jpg
        frame_path = os.path.join(frames_dir, frame_name)
        if not os.path.exists(frame_path):
            frame_path = os.path.join(frames_dir, frame_name + '.jpg')
        if not os.path.exists(frame_path):
            print(f"Warning: Frame not found at {frame_path}")
            continue
        
        img = np.array(Image.open(frame_path).convert('RGB'))
        
        # 加载GT mask
        gt_mask = None
        if frame_name in gt_masks_rle and gt_masks_rle[frame_name] is not None:
            gt_mask = rle2mask(gt_masks_rle[frame_name])
        
        # 加载预测mask
        pred_mask = None
        if frame_name in pred_masks_rle and pred_masks_rle[frame_name] is not None:
            pred_mask = rle2mask(pred_masks_rle[frame_name])
        
        # 第一行：原始图像
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f'Frame: {frame_name}', fontsize=8)
        axes[0, idx].axis('off')
        
        # 第二行：GT mask叠加
        if gt_mask is not None:
            img_with_gt = overlay_mask(img, gt_mask, color=[0, 255, 0], alpha=0.5)
            axes[1, idx].imshow(img_with_gt)
            axes[1, idx].set_title('GT (Green)', fontsize=8, color='green')
        else:
            axes[1, idx].imshow(img)
            axes[1, idx].set_title('GT: None', fontsize=8, color='gray')
        axes[1, idx].axis('off')
        
        # 第三行：预测mask叠加
        if pred_mask is not None:
            img_with_pred = overlay_mask(img, pred_mask, color=[255, 0, 0], alpha=0.5)
            axes[2, idx].imshow(img_with_pred)
            
            # 计算IoU
            if gt_mask is not None:
                overlap = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou = overlap / union if union > 0 else 0.0
                axes[2, idx].set_title(f'Pred (Red)\nIoU: {iou:.2f}', fontsize=8, color='red')
            else:
                axes[2, idx].set_title('Pred (Red)\n(False Positive)', fontsize=8, color='orange')
        else:
            axes[2, idx].imshow(img)
            if gt_mask is not None:
                axes[2, idx].set_title('Pred: None\n(False Negative)', fontsize=8, color='purple')
            else:
                axes[2, idx].set_title('Pred: None', fontsize=8, color='gray')
        axes[2, idx].axis('off')
    
    # 添加总标题
    fig.suptitle(f'Video: {video_id}\nExpression {exp_id} (obj_id={obj_id}): "{expression}"', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{video_id}_exp{exp_id}_obj{obj_id}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_file", type=str, default='/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/egomask/meta_expressions.json'
                        help="meta_expressions.json文件路径")
    parser.add_argument("--frames_root", type=str, default='/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/EgoMask/dataset/egomask/JPEGImages',
                        help="视频帧根目录")
    parser.add_argument("--gt_masks_root", type=str, required=True,
                        help="GT masks根目录")
    parser.add_argument("--pred_masks_root", type=str, required=True,
                        help="预测masks根目录")
    parser.add_argument("--output_dir", type=str, default="./egomask_visualizations",
                        help="输出目录")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="可视化多少个样本")
    parser.add_argument("--max_frames", type=int, default=8,
                        help="每个样本最多可视化多少帧")
    parser.add_argument("--min_iou", type=float, default=0.0,
                        help="最小IoU阈值，只可视化IoU>该值的样本")
    args = parser.parse_args()

    # 加载meta
    print(f"Loading meta from {args.meta_file}...")
    with open(args.meta_file) as f:
        meta = json.load(f)

    videos = meta['videos']
    print(f"Found {len(videos)} videos")

    # 收集所有样本并计算IoU
    print(f"Collecting samples with IoU > {args.min_iou}...")
    samples = []
    for video_id, video_info in videos.items():
        for exp_id, exp_info in video_info['expressions'].items():
            obj_id = exp_info['obj_id']

            # 构建路径
            gt_mask_path = os.path.join(args.gt_masks_root, video_id, f"{obj_id}.json")
            pred_mask_path = os.path.join(args.pred_masks_root, video_id, exp_id, f"{exp_id}-{obj_id}.json")

            # 检查预测文件是否存在
            if not os.path.exists(pred_mask_path):
                continue

            # 加载masks并计算平均IoU
            try:
                with open(gt_mask_path) as f:
                    gt_masks = json.load(f)
                with open(pred_mask_path) as f:
                    pred_masks = json.load(f)

                ious = []
                for frame_name in video_info['frames']:
                    gt_rle = gt_masks.get(frame_name)
                    pred_rle = pred_masks.get(frame_name)

                    # 跳过两者都没有mask的帧
                    if gt_rle is None and pred_rle is None:
                        continue

                    # 转换为mask
                    gt_mask = rle2mask(gt_rle) if gt_rle is not None else None
                    pred_mask = rle2mask(pred_rle) if pred_rle is not None else None

                    # 计算IoU
                    if gt_mask is not None and pred_mask is not None:
                        overlap = np.logical_and(gt_mask, pred_mask).sum()
                        union = np.logical_or(gt_mask, pred_mask).sum()
                        iou = overlap / union if union > 0 else 0.0
                        ious.append(iou)

                avg_iou = np.mean(ious) if len(ious) > 0 else 0.0

                # 只保留IoU > threshold的样本
                if avg_iou > args.min_iou:
                    samples.append({
                        'video_id': video_id,
                        'exp_id': exp_id,
                        'obj_id': obj_id,
                        'expression': exp_info['exp'],
                        'frames': video_info['frames'],
                        'avg_iou': avg_iou,
                    })
            except Exception as e:
                print(f"Warning: Failed to process {video_id}/{exp_id}: {e}")
                continue

    print(f"Found {len(samples)} samples with IoU > {args.min_iou}")

    # 按IoU降序排序
    samples.sort(key=lambda x: x['avg_iou'], reverse=True)

    # 限制样本数
    if len(samples) > args.num_samples:
        samples = samples[:args.num_samples]
    
    print(f"Visualizing {len(samples)} samples...")
    
    # 可视化每个样本
    for idx, sample in enumerate(samples):
        avg_iou = sample.get('avg_iou', 0.0)
        print(f"\n[{idx+1}/{len(samples)}] Processing {sample['video_id']} exp{sample['exp_id']} (avg_iou={avg_iou:.3f})...")

        video_id = sample['video_id']
        exp_id = sample['exp_id']
        obj_id = sample['obj_id']
        expression = sample['expression']
        
        # 构建路径
        frames_dir = os.path.join(args.frames_root, video_id)
        gt_mask_path = os.path.join(args.gt_masks_root, video_id, f"{obj_id}.json")
        pred_mask_path = os.path.join(args.pred_masks_root, video_id, exp_id, f"{exp_id}-{obj_id}.json")
        
        # 可视化
        try:
            visualize_sample(
                video_id, exp_id, obj_id,
                frames_dir, gt_mask_path, pred_mask_path,
                expression, args.output_dir, args.max_frames
            )
        except Exception as e:
            print(f"Error processing sample: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Done! Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()

