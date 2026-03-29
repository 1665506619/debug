"""
可视化GT和Pred的对比
基于eval_egomask.py的逻辑，展示每个样本的GT mask和Pred mask
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
import json
import argparse
import numpy as np
import cv2
import pycocotools.mask as maskUtils
from collections import defaultdict

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def visualize_sample_v2(video_id, orig_video_id, exp_id, obj_id, vid_info, gt_dir, pred_path, image_base_dir, output_dir, max_frames=10):
    """
    可视化单个样本的GT和Pred对比
    video_id: 完整的video_id（包含--timestamp-duration）
    orig_video_id: 原始video_id（不包含--timestamp-duration）
    """

    exp_name = f"{video_id}_{exp_id}_{obj_id}"

    # 检查预测是否存在
    if not os.path.exists(f"{pred_path}/{video_id}"):
        print(f"  预测不存在: {video_id}")
        return None

    # 加载GT masks（使用完整video_id）
    gt_file = os.path.join(gt_dir, f"{video_id}/{obj_id}.json")
    if not os.path.exists(gt_file):
        print(f"  GT文件不存在: {gt_file}")
        return None
    gold_mask_rle = read_json(gt_file)

    # 加载Pred masks（使用完整video_id）
    pred_file = os.path.join(pred_path, f"{video_id}/{exp_id}/{exp_id}-{obj_id}.json")
    if not os.path.exists(pred_file):
        print(f"  预测文件不存在: {pred_file}")
        return None

    pred_mask_rle = read_json(pred_file)
    
    # 获取图像尺寸
    h, w = None, None
    for fname, mask_rle in gold_mask_rle.items():
        mask = maskUtils.decode(mask_rle)
        if mask is not None:
            h, w = mask.shape
            break
    
    if h is None:
        print(f"  无法获取图像尺寸")
        return None
    
    vid_len = len(vid_info["frames"])
    
    # 统计信息
    stats = {
        'TT': 0,  # GT有 + Pred有
        'TF': 0,  # GT有 + Pred无
        'FT': 0,  # GT无 + Pred有
        'FF': 0,  # GT无 + Pred无
        'gt_frames': [],
        'pred_frames': [],
        'iou_list': [],
    }
    
    # 收集帧信息
    frame_info = []
    for fidx, fname in enumerate(vid_info["frames"]):
        has_gt = fname in gold_mask_rle
        has_pred = fname in pred_mask_rle
        
        # 计算IoU
        iou = 0.0
        if has_gt or has_pred:
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            
            if has_gt:
                gt_mask = maskUtils.decode(gold_mask_rle[fname])
            if has_pred:
                pred_mask = maskUtils.decode(pred_mask_rle[fname])
            
            overlap = np.logical_and(gt_mask, pred_mask)
            union = np.logical_or(gt_mask, pred_mask)
            
            if union.sum() > 0:
                iou = overlap.sum() / union.sum()
            else:
                iou = 1.0
        else:
            iou = 1.0
        
        # 统计类型
        if has_gt and has_pred:
            stats['TT'] += 1
            frame_type = 'TT'
        elif has_gt and not has_pred:
            stats['TF'] += 1
            frame_type = 'TF'
        elif not has_gt and has_pred:
            stats['FT'] += 1
            frame_type = 'FT'
        else:
            stats['FF'] += 1
            frame_type = 'FF'
        
        if has_gt:
            stats['gt_frames'].append(fidx)
        if has_pred:
            stats['pred_frames'].append(fidx)
        
        stats['iou_list'].append(iou)
        
        frame_info.append({
            'fidx': fidx,
            'fname': fname,
            'has_gt': has_gt,
            'has_pred': has_pred,
            'iou': iou,
            'type': frame_type,
        })
    
    # 选择要可视化的帧（优先选择TF和TT类型）
    tf_frames = [f for f in frame_info if f['type'] == 'TF']
    tt_frames = [f for f in frame_info if f['type'] == 'TT']
    ft_frames = [f for f in frame_info if f['type'] == 'FT']
    
    # 选择帧：优先TF，然后TT，然后FT
    selected_frames = []
    selected_frames.extend(tf_frames[:max_frames//2])
    selected_frames.extend(tt_frames[:max_frames//2])
    if len(selected_frames) < max_frames:
        selected_frames.extend(ft_frames[:max_frames-len(selected_frames)])
    
    if len(selected_frames) == 0:
        print(f"  没有可视化的帧")
        return stats
    
    # 限制数量
    selected_frames = selected_frames[:max_frames]
    
    # 创建可视化
    vis_images = []
    for frame in selected_frames:
        fidx = frame['fidx']
        fname = frame['fname']
        
        # 加载图像（使用完整video_id）
        img_path = os.path.join(image_base_dir, f"{video_id}/{fname}.jpg")
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 创建三列：原图、GT、Pred
        img_orig = img.copy()
        img_gt = img.copy()
        img_pred = img.copy()
        
        # 叠加GT mask (绿色)
        if frame['has_gt']:
            gt_mask = maskUtils.decode(gold_mask_rle[fname])
            img_gt[gt_mask > 0] = (img_gt[gt_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        
        # 叠加Pred mask (蓝色)
        if frame['has_pred']:
            pred_mask = maskUtils.decode(pred_mask_rle[fname])
            img_pred[pred_mask > 0] = (img_pred[pred_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        
        # 添加标题
        cv2.putText(img_orig, f'Frame {fidx}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_gt, f'GT: {"YES" if frame["has_gt"] else "NO"}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if frame["has_gt"] else (128, 128, 128), 2)
        cv2.putText(img_pred, f'Pred: {"YES" if frame["has_pred"] else "NO"}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0) if frame["has_pred"] else (128, 128, 128), 2)
        
        # 添加类型和IoU
        type_color = {
            'TT': (0, 255, 0),
            'TF': (0, 0, 255),
            'FT': (0, 165, 255),
            'FF': (128, 128, 128),
        }
        cv2.putText(img_orig, f'{frame["type"]} IoU:{frame["iou"]*100:.1f}%', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, type_color[frame['type']], 2)
        
        # 拼接三列
        combined = np.hstack([img_orig, img_gt, img_pred])
        vis_images.append(combined)
    
    if len(vis_images) == 0:
        print(f"  没有成功加载的图像")
        return stats
    
    # 垂直拼接所有帧
    final_img = np.vstack(vis_images)
    
    # 添加总标题
    expression = vid_info['expressions'][exp_id].get('exp', 'Unknown')
    title_height = 100
    title_img = np.ones((title_height, final_img.shape[1], 3), dtype=np.uint8) * 255
    
    cv2.putText(title_img, f'{exp_name}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(title_img, f'Expression: {expression[:100]}', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(title_img, f'TT:{stats["TT"]} TF:{stats["TF"]} FT:{stats["FT"]} FF:{stats["FF"]} | Avg IoU:{np.mean(stats["iou_list"])*100:.1f}%', 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    final_img = np.vstack([title_img, final_img])
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{exp_name}.jpg')
    cv2.imwrite(output_path, final_img)
    
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='short', choices=['long', 'medium', 'short', 'full'])
    parser.add_argument('--pred_path', type=str, required=True, help='预测结果路径')
    parser.add_argument('--question_file', type=str, required=True, help='question文件路径（egomask_short.json）')
    parser.add_argument('--gt_dir', type=str, required=True, help='GT annotations目录')
    parser.add_argument('--image_base_dir', type=str, required=True, help='图片基础目录')
    parser.add_argument('--output_dir', type=str, default='gt_pred_vis', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=20, help='可视化样本数')
    parser.add_argument('--max_frames', type=int, default=6, help='每个样本最多可视化多少帧')
    parser.add_argument('--result_file', type=str, default=None, help='结果文件路径（用于筛选高TF样本）')
    parser.add_argument('--filter_mode', type=str, default='first',
                       choices=['first', 'high_tf', 'low_iou'],
                       help='筛选模式：first=前N个, high_tf=高TF比例, low_iou=低IoU')

    args = parser.parse_args()

    # 加载question文件
    with open(args.question_file, 'r') as f:
        questions = json.load(f)

    print(f'总样本数: {len(questions)}')

    # 如果需要筛选，加载结果文件
    if args.filter_mode != 'first' and args.result_file:
        print(f'\n加载结果文件: {args.result_file}')
        with open(args.result_file, 'r') as f:
            results = [json.loads(line) for line in f]

        # 创建结果字典
        result_dict = {r['seq_exp']: r for r in results}

        # 计算每个样本的TF比例
        sample_scores = []
        for q in questions:
            video_id = q['video_id']
            exp_id = q['exp_id']
            obj_id = q['obj_id']
            seq_exp = f"{video_id}_{exp_id}_{obj_id}"

            if seq_exp in result_dict:
                r = result_dict[seq_exp]

                if args.filter_mode == 'high_tf':
                    # 计算TF比例
                    gt_temporal = r['T']['gt_temporal']
                    res_temporal = r['T']['res_temporal']

                    tf_count = sum(1 for g, p in zip(gt_temporal, res_temporal) if g and not p)
                    gt_count = sum(gt_temporal)

                    if gt_count > 0:
                        tf_ratio = tf_count / gt_count
                    else:
                        tf_ratio = 0.0

                    sample_scores.append((tf_ratio, q))

                elif args.filter_mode == 'low_iou':
                    # 使用gold_iou排序
                    iou = r['gold_iou']
                    sample_scores.append((iou, q))

        # 排序
        if args.filter_mode == 'high_tf':
            # TF比例从高到低
            sample_scores.sort(key=lambda x: x[0], reverse=True)
            print(f'\n按TF比例排序（从高到低）')
            for i, (score, q) in enumerate(sample_scores[:10]):
                print(f'  {i+1}. {q["video_id"]}_{q["exp_id"]}_{q["obj_id"]}: TF比例={score*100:.1f}%')

        elif args.filter_mode == 'low_iou':
            # IoU从低到高
            sample_scores.sort(key=lambda x: x[0])
            print(f'\n按IoU排序（从低到高）')
            for i, (score, q) in enumerate(sample_scores[:10]):
                print(f'  {i+1}. {q["video_id"]}_{q["exp_id"]}_{q["obj_id"]}: IoU={score*100:.1f}%')

        # 使用筛选后的样本
        questions = [q for _, q in sample_scores]

    # 可视化样本
    print(f'\n开始可视化前{args.num_samples}个样本...')

    all_stats = []
    success_count = 0

    for i, q in enumerate(questions[:args.num_samples]):
        video_id = q['video_id']
        exp_id = q['exp_id']
        obj_id = q['obj_id']
        expression = q['expression']
        frame_names = q['frame_names']

        # 从video_id中提取原始video_id（去掉--timestamp-duration部分）
        parts = video_id.split('--')
        if len(parts) == 2:
            orig_video_id = parts[0]
        else:
            orig_video_id = video_id

        print(f'\n样本 {i+1}/{args.num_samples}: {video_id}_{exp_id}_{obj_id}')

        # 构建vid_info
        vid_info = {
            'frames': frame_names,
            'expressions': {
                exp_id: {
                    'exp': expression,
                    'obj_id': obj_id
                }
            }
        }

        # GT路径使用原始video_id
        gt_dir = args.gt_dir
        # 图片路径使用完整video_id
        image_dir = args.image_base_dir

        stats = visualize_sample_v2(
            video_id, orig_video_id, exp_id, obj_id, vid_info,
            gt_dir, args.pred_path, image_dir,
            args.output_dir, args.max_frames
        )
        
        if stats:
            all_stats.append(stats)
            success_count += 1
            print(f'  保存成功: {args.output_dir}/{video_id}_{exp_id}_{obj_id}.jpg')
            print(f'  统计: TT={stats["TT"]}, TF={stats["TF"]}, FT={stats["FT"]}, FF={stats["FF"]}')
            print(f'  平均IoU: {np.mean(stats["iou_list"])*100:.1f}%')
    
    print(f'\n可视化完成！')
    print(f'成功: {success_count}/{args.num_samples}')
    print(f'输出目录: {args.output_dir}')
    
    # 打印总体统计
    if all_stats:
        total_TT = sum(s['TT'] for s in all_stats)
        total_TF = sum(s['TF'] for s in all_stats)
        total_FT = sum(s['FT'] for s in all_stats)
        total_FF = sum(s['FF'] for s in all_stats)
        total = total_TT + total_TF + total_FT + total_FF
        
        print(f'\n总体统计:')
        print(f'  TT: {total_TT} ({total_TT/total*100:.1f}%)')
        print(f'  TF: {total_TF} ({total_TF/total*100:.1f}%)')
        print(f'  FT: {total_FT} ({total_FT/total*100:.1f}%)')
        print(f'  FF: {total_FF} ({total_FF/total*100:.1f}%)')

