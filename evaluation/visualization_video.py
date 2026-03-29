import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
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


def mask_to_color(mask):
    """Convert a mask to a colored image."""
    arr = np.array(mask)
    color = np.zeros((*arr.shape, 3), dtype=np.uint8)
    color[arr == 1] = [255, 0, 0]    # Class 1 -> Red
    color[arr == 2] = [0, 255, 0]    # Class 2 -> Green
    color[arr == 3] = [0, 0, 255]    # Class 3 -> Blue
    return color


def overlay(img, mask_color, alpha=0.5):
    """Overlay the mask on the image with alpha blending."""
    img_f = img.astype(np.float32)
    mask_f = mask_color.astype(np.float32)
    return (img_f * (1 - alpha) + mask_f * alpha).astype(np.uint8)


def visualize_video_frames(video_path, gt_masks, pred_masks, frames, caption, save_path, j_f):
    """
    Visualize multiple sampled frames from a video in a single image.
    Each sample includes both the ground truth and prediction overlays.
    """
    # 均匀采样16帧
    sampled_frames_indices = np.linspace(0, len(frames) - 1, 16, dtype=int)
    sampled_frames = [frames[i] for i in sampled_frames_indices]
    
    # 创建 4行 × 8列 画布
    fig, axs = plt.subplots(4, 8, figsize=(28, 14))  # 4行 (GT + Pred), 8列 (每行8个采样帧)
    
    for idx, frame_idx in enumerate(sampled_frames_indices):
        # 当前帧在子图中的行列位置 (前2行是GT，后2行是Pred)
        row_gt = idx // 8  # 计算GT行索引 (0~1)
        col_gt = idx % 8   # 计算列索引 (0~7)
        row_pred = row_gt + 2  # Pred行索引与GT行相差2
        
        # 加载当前帧图像
        if isinstance(video_path, list):
            img_path = video_path[sampled_frames_indices[idx]]
        else:
            img_path = os.path.join(video_path, sampled_frames[idx])
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # 加载并处理当前帧的真值和预测掩码
        gt_mask = gt_masks[frame_idx]
        pred_mask = pred_masks[frame_idx]
        color_gt = mask_to_color(gt_mask)
        color_pred = mask_to_color(pred_mask)
        
        # 创建Overlay图像
        overlay_gt = overlay(img_np, color_gt, alpha=0.5)
        overlay_pred = overlay(img_np, color_pred, alpha=0.5)
        
        # 显示Ground Truth Overlay到子图
        axs[row_gt, col_gt].imshow(overlay_gt)
        axs[row_gt, col_gt].set_title(f"GT Frame {frame_idx}", fontsize=10)
        
        # 显示Prediction Overlay到子图
        axs[row_pred, col_gt].imshow(overlay_pred)
        axs[row_pred, col_gt].set_title(f"Pred Frame {frame_idx}", fontsize=10)
        
        # 隐藏坐标轴
        axs[row_gt, col_gt].axis('off')
        axs[row_pred, col_gt].axis('off')
        
    plt.tight_layout(rect=[0, 0.07, 1, 0.92])  # 为标题和caption留空间
    plt.suptitle(f"J&F: {j_f:.4f}", fontsize=18, color='blue')
    # 增加caption到图片底部
    fig.subplots_adjust(bottom=0.15)  # 为caption留出空间
    fig.text(0.5, 0.04, f"Caption: {caption}", wrap=True, fontsize=14, ha='center', color='black')
    
    # 保存可视化结果
    plt.tight_layout(rect=[0, 0.07, 1, 1])  # 改变图像布局以适应标题和caption
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/code/video-seg/evaluation_results/1127_v2_lora/checkpoint-8623/egomask_short.json')
    parser.add_argument('--gt_json', type=str, default='/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_short.json')
    parser.add_argument('--save', type=str, default='visualization/egomask/')
    args = parser.parse_args()
    
    with open(args.json, 'r', encoding='utf8') as f:
        pred_datas = json.load(f)
    
    with open(args.gt_json, 'r', encoding='utf8') as f:
        gt_datas = json.load(f)
    
    os.makedirs(args.save, exist_ok=True)

    for pred_data in pred_datas:
        if 'idx' not in pred_data:
            continue
        idx = pred_data['idx']
        gt_data = gt_datas[idx]
        video_path = pred_data['video_path']
        if isinstance(video_path, list):
            frame_list = video_path
        else:
            frame_list = sorted(os.listdir(video_path))  # 获取视频中的帧列表

        gt_masks = [annToMask(gt_data['masks'][i]) if gt_data['masks'][i] is not None else None for i in range(len(frame_list))]
        pred_masks = [annToMask(pred_data['mask_rle'][i]) for i in range(len(frame_list))]
        
        # 生成保存路径
        save_path = os.path.join(args.save, f"video_{pred_data['idx']}.jpg")
        
        # 获取caption信息
        caption = pred_data.get('instruction', 'No caption provided.')
        j_f = (pred_data['j']+pred_data['f'])/2
        
        # 生成可视化
        visualize_video_frames(video_path, gt_masks, pred_masks, frame_list, caption, save_path, j_f)
        print(f"Saved visualization: {save_path}")


if __name__ == '__main__':
    main()
