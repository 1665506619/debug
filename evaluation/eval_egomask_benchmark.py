"""
EgoMask Benchmark 评估脚本
完全按照官方评估方式计算 iou_overall, iou_gold, iou_gold_with_pred

文件结构:
- 预测文件: {pred_path}/short/{video_id}/{exp_id}/{exp_id}-{obj_id}.json
- GT文件: {annotation_root}/{video_id}/{obj_id}.json
- Question文件: {question_file} (包含video_id, exp_id, obj_id等信息)
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
import json
import time
import argparse
import numpy as np
import pycocotools.mask as maskUtils

# 添加当前目录到sys.path以支持相对导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp


# ============================================================================
# 从metrics.py复制的函数（避免导入torch）
# ============================================================================

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.
    """
    import math

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can't convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute F-measure for boundaries between foreground_mask and gt_mask.
    """
    import cv2

    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    # 使用cv2.dilate代替skimage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(bound_pix)*2+1, int(bound_pix)*2+1))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), kernel)
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), kernel)

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    """Compute boundary F-measure."""
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def db_eval_boundary_temporal(gt_masks, pred_masks):
    """
    计算时序边界指标
    返回包含accuracy, precision, recall, f1的字典
    """
    # 简化实现：基于帧级别的检测
    n_frames = len(gt_masks)

    gt_temporal = []
    res_temporal = []

    for i in range(n_frames):
        gt_has_obj = np.sum(gt_masks[i]) > 0
        pred_has_obj = np.sum(pred_masks[i]) > 0

        gt_temporal.append(1 if gt_has_obj else 0)
        res_temporal.append(1 if pred_has_obj else 0)

    gt_temporal = np.array(gt_temporal)
    res_temporal = np.array(res_temporal)

    # 计算TP, FP, FN, TN
    tp = np.sum((gt_temporal == 1) & (res_temporal == 1))
    fp = np.sum((gt_temporal == 0) & (res_temporal == 1))
    fn = np.sum((gt_temporal == 1) & (res_temporal == 0))
    tn = np.sum((gt_temporal == 0) & (res_temporal == 0))

    # 计算指标
    accuracy = (tp + tn) / n_frames if n_frames > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_temporal": gt_temporal.tolist(),
        "res_temporal": res_temporal.tolist(),
    }


NUM_WORKERS = 32


def read_json(path):
    """读取JSON文件"""
    with open(path, 'r') as f:
        return json.load(f)


def write_json(data, path):
    """写入JSON文件"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def write_jsonl(data, path):
    """写入JSONL文件"""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def is_empty_mask(mask_rle):
    """
    判断mask是否为空
    空mask的特征: counts为特定的空字符串模式
    """
    if mask_rle is None:
        return True
    if 'counts' not in mask_rle:
        return True
    counts = mask_rle['counts']
    # 空mask的常见模式
    empty_patterns = ['PP\\9', 'PPQ7', 'PP\\\\9', 'PP9', '']
    return counts in empty_patterns or len(counts) == 0


def eval_single_video(video_id, exp_id, obj_id, pred_path, annotation_root, frame_names):
    """
    评估单个视频的单个expression
    
    Args:
        video_id: 视频ID
        exp_id: Expression ID
        obj_id: 物体ID
        pred_path: 预测结果根目录
        annotation_root: GT标注根目录
        frame_names: 帧名称列表
    
    Returns:
        dict: 包含各项指标的字典
    """
    exp_name = f"{video_id}_{exp_id}_{obj_id}"
    
    # 构建文件路径
    pred_file = os.path.join(pred_path, video_id, str(exp_id), f"{exp_id}-{obj_id}.json")
    gt_file = os.path.join(annotation_root, video_id, f"{obj_id}.json")
    
    # 检查文件是否存在
    if not os.path.exists(pred_file):
        print(f"Warning: Prediction file not found: {pred_file}")
        return None
    
    if not os.path.exists(gt_file):
        print(f"Warning: GT file not found: {gt_file}")
        return None
    
    # 加载预测和GT masks
    try:
        pred_mask_rle = read_json(pred_file)
        gt_mask_rle = read_json(gt_file)
    except Exception as e:
        print(f"Error loading files for {exp_name}: {e}")
        return None
    
    # 获取图像尺寸
    h, w = None, None
    for fname, mask_rle in gt_mask_rle.items():
        if mask_rle is not None and not is_empty_mask(mask_rle):
            try:
                mask = maskUtils.decode(mask_rle)
                if mask is not None:
                    h, w = mask.shape
                    break
            except:
                continue
    
    if h is None:
        # 尝试从预测中获取尺寸
        for fname, mask_rle in pred_mask_rle.items():
            if mask_rle is not None and 'size' in mask_rle:
                h, w = mask_rle['size']
                break
    
    if h is None:
        print(f"Warning: Cannot determine image size for {exp_name}")
        return None
    
    # 构建mask数组
    vid_len = len(frame_names)
    gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
    pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
    
    # 填充GT和预测masks
    for fidx, fname in enumerate(frame_names):
        # 填充GT mask
        if fname in gt_mask_rle:
            gt_segm = gt_mask_rle[fname]
            if gt_segm is not None and not is_empty_mask(gt_segm):
                try:
                    mask = maskUtils.decode(gt_segm)
                    if mask is not None and mask.shape == (h, w):
                        gt_masks[fidx, :, :] = np.array(mask, dtype=np.uint8)
                except:
                    pass
        
        # 填充预测mask
        if fname in pred_mask_rle:
            pred_segm = pred_mask_rle[fname]
            if pred_segm is not None and not is_empty_mask(pred_segm):
                try:
                    mask = maskUtils.decode(pred_segm)
                    if mask is not None and mask.shape == (h, w):
                        pred_masks[fidx, :, :] = np.array(mask, dtype=np.uint8)
                except:
                    pass
    
    # 计算IoU指标（完全按照官方方式）
    overall_iou_list = []      # for iou_overall
    gold_iou_list = []          # for iou_gold
    gold_with_pred_iou_list = []  # for iou_gold_with_pred
    
    for fidx in range(vid_len):
        # 计算overlap和union
        overlap = np.logical_and(gt_masks[fidx], pred_masks[fidx])
        union = np.logical_or(gt_masks[fidx], pred_masks[fidx])
        
        if union.sum() > 0:
            iou_ = overlap.sum() / union.sum()
            overall_iou_list.append(iou_)
        else:
            # no gt mask & no pred mask
            iou_ = 1.0
            overall_iou_list.append(iou_)
        
        # 检查当前帧是否有GT
        fname = frame_names[fidx]
        has_gt = fname in gt_mask_rle and gt_mask_rle[fname] is not None and not is_empty_mask(gt_mask_rle[fname])
        has_pred = fname in pred_mask_rle and pred_mask_rle[fname] is not None and not is_empty_mask(pred_mask_rle[fname])
        
        if has_gt:
            # GT非空: 加入gold_iou_list和gold_with_pred_iou_list
            gold_iou_list.append(iou_)
            gold_with_pred_iou_list.append(iou_)
        elif has_pred:
            # 只有预测非空，GT为空: 只加入gold_with_pred_iou_list (惩罚误报)
            gold_with_pred_iou_list.append(iou_)
    
    # 计算J, F, T指标
    j = db_eval_iou(gt_masks, pred_masks).mean()
    f = db_eval_boundary(gt_masks, pred_masks).mean()
    t = db_eval_boundary_temporal(gt_masks, pred_masks)
    
    # 计算平均IoU
    overall_iou = np.mean(overall_iou_list) if overall_iou_list else 0.0
    gold_iou = np.mean(gold_iou_list) if gold_iou_list else 0.0
    gold_with_pred_iou = np.mean(gold_with_pred_iou_list) if gold_with_pred_iou_list else 0.0
    
    return {
        "J": j,
        "F": f,
        "T": t,
        "overall_iou": overall_iou,
        "gold_iou": gold_iou,
        "gold_with_pred_iou": gold_with_pred_iou,
        "video_id": video_id,
        "exp_id": str(exp_id),
        "obj_id": obj_id,
    }


def eval_queue(q, rank, output_dict, pred_path, annotation_root):
    """多进程评估队列"""
    while not q.empty():
        try:
            video_id, exp_id, obj_id, frame_names = q.get(timeout=1)
            exp_name = f"{video_id}_{exp_id}_{obj_id}"
            
            result = eval_single_video(video_id, exp_id, obj_id, pred_path, annotation_root, frame_names)
            
            if result is not None:
                output_dict[exp_name] = result
        except:
            break


def main():
    parser = argparse.ArgumentParser(description="EgoMask Benchmark Evaluation")
    parser.add_argument("--dataset_type", type=str, default="short", 
                        choices=["long", "medium", "short", "full"],
                        help="数据集类型")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="预测结果根目录 (包含short/medium/long子目录)")
    parser.add_argument("--question_file", type=str, required=True,
                        help="Question文件路径 (包含video_id, exp_id, obj_id等信息)")
    parser.add_argument("--annotation_root", type=str, required=True,
                        help="GT标注根目录")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认为pred_path)")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="并行进程数")
    parser.add_argument("--save_name", type=str, default="eval_results.json",
                        help="输出文件名")
    args = parser.parse_args()
    
    # 设置路径
    pred_path = os.path.join(args.pred_path, args.dataset_type)
    if args.output_dir is None:
        args.output_dir = pred_path
    
    # 加载question文件
    print(f"Loading questions from {args.question_file}...")
    questions = read_json(args.question_file)
    print(f"Loaded {len(questions)} questions")
    
    # 创建评估队列
    queue = mp.Queue()
    output_dict = mp.Manager().dict()
    
    for q in questions:
        video_id = q['video_id']
        exp_id = q['exp_id']
        obj_id = q['obj_id']
        frame_names = q['frame_names']
        queue.put([video_id, exp_id, obj_id, frame_names])
    
    cnt = queue.qsize()
    print(f"Total samples to evaluate: {cnt}")
    
    # 多进程评估
    start_time = time.time()
    processes = []
    for rank in range(args.num_workers):
        p = mp.Process(
            target=eval_queue,
            args=(queue, rank, output_dict, pred_path, args.annotation_root)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print(f"\nEvaluated {len(output_dict)}/{cnt} samples")
    
    if len(output_dict) == 0:
        print("Error: No valid results!")
        return
    
    if cnt != len(output_dict):
        print(f"Warning: Expected {cnt} samples, got {len(output_dict)} (missing {cnt - len(output_dict)})")
    
    # 汇总结果
    j = [output_dict[x]["J"] for x in output_dict]
    f = [output_dict[x]["F"] for x in output_dict]
    t_f1 = [output_dict[x]["T"]["f1"] for x in output_dict]
    t_acc = [output_dict[x]["T"]["accuracy"] for x in output_dict]
    t_precision = [output_dict[x]["T"]["precision"] for x in output_dict]
    t_recall = [output_dict[x]["T"]["recall"] for x in output_dict]
    
    iou_overall = [output_dict[x]["overall_iou"] for x in output_dict]
    iou_gold = [output_dict[x]["gold_iou"] for x in output_dict]
    iou_gold_with_pred = [output_dict[x]["gold_with_pred_iou"] for x in output_dict]
    
    # 计算汇总指标
    results = {
        "J": round(100 * float(np.mean(j)), 2),
        "F": round(100 * float(np.mean(f)), 2),
        "J&F": round(100 * float((np.mean(j) + np.mean(f)) / 2), 2),
        "T_f1": round(100 * float(np.mean(t_f1)), 2),
        "T_acc": round(100 * float(np.mean(t_acc)), 2),
        "T_precision": round(100 * float(np.mean(t_precision)), 2),
        "T_recall": round(100 * float(np.mean(t_recall)), 2),
        "iou_overall": round(100 * float(np.mean(iou_overall)), 2),
        "iou_gold": round(100 * float(np.mean(iou_gold)), 2),
        "iou_gold_with_pred": round(100 * float(np.mean(iou_gold_with_pred)), 2),
        "num_samples": len(output_dict),
    }
    
    # 打印结果
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for key, value in results.items():
        print(f"{key:20s}: {value}")
    print("="*80)
    
    # 保存结果
    output_path = os.path.join(args.output_dir, args.save_name)
    write_json(results, output_path)
    print(f"\nResults saved to {output_path}")
    
    # 保存详细结果
    output_jsonl = [dict(seq_exp=k, **v) for k, v in output_dict.items()]
    jsonl_path = output_path.replace(".json", "_full_result.jsonl")
    write_jsonl(output_jsonl, jsonl_path)
    print(f"Detailed results saved to {jsonl_path}")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")


if __name__ == "__main__":
    main()

