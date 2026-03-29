import json
import os
import numpy as np


def postprocess_seg_result(results):
    """
    Post-process segmentation results to compute aggregated metrics.
    """
    if not results:
        return results
    
    # Calculate average metrics
    total_iou = sum(d.get('iou', 0) for d in results)
    total_iou_non_zero = sum(d.get('iou_non_zero', 0) for d in results if 'iou_non_zero' in d)
    total_iou_zero = sum(d.get('iou_zero', 0) for d in results if 'iou_zero' in d)
    total_f = sum(d.get('f', 0) for d in results)
    
    count_iou_non_zero = sum(1 for d in results if 'iou_non_zero' in d)
    count_iou_zero = sum(1 for d in results if 'iou_zero' in d)
    
    # Calculate cls accuracy if cls_scores exist
    cls_correct = 0
    cls_total = 0
    for result in results:
        if 'cls_scores' in result:
            cls_scores = result['cls_scores']
            if isinstance(cls_scores, list):
                # For video: check if predictions match ground truth existence
                # Assume if any frame has valid prediction (cls_score > 0), it's positive
                cls_total += 1
                # Simple heuristic: if model predicts positive and IoU > 0, it's correct
                if result.get('iou', 0) > 0:
                    cls_correct += 1
    
    # Aggregate metrics by type
    type_metrics = {}
    for result in results:
        task_type = result.get('type', 'unknown')
        if task_type not in type_metrics:
            type_metrics[task_type] = {'iou': [], 'iou_non_zero': [], 'iou_zero': [], 'f': []}
        
        if 'iou' in result:
            type_metrics[task_type]['iou'].append(result['iou'])
        if 'iou_non_zero' in result:
            type_metrics[task_type]['iou_non_zero'].append(result['iou_non_zero'])
        if 'iou_zero' in result:
            type_metrics[task_type]['iou_zero'].append(result['iou_zero'])
        if 'f' in result:
            type_metrics[task_type]['f'].append(result['f'])
    
    # Calculate average metrics per type
    summary = {
        'overall': {
            'avg_iou': total_iou / len(results) if results else 0,
            'avg_iou_non_zero': total_iou_non_zero / count_iou_non_zero if count_iou_non_zero > 0 else 0,
            'avg_iou_zero': total_iou_zero / count_iou_zero if count_iou_zero > 0 else 0,
            'avg_f': total_f / len(results) if results else 0,
            'j&f': (total_iou + total_f) / (2 * len(results)) if results else 0,
            'cls_accuracy': cls_correct / cls_total if cls_total > 0 else 0,
            'num_samples': len(results)
        },
        'by_type': {}
    }
    
    for task_type, metrics in type_metrics.items():
        avg_j = np.mean(metrics['iou']) if metrics['iou'] else 0
        avg_f = np.mean(metrics['f']) if metrics['f'] else 0
        summary['by_type'][task_type] = {
            'j': avg_j,
            'f': avg_f,
            'j&f': (avg_j + avg_f) / 2,
            'avg_iou_non_zero': np.mean(metrics['iou_non_zero']) if metrics['iou_non_zero'] else 0,
            'avg_iou_zero': np.mean(metrics['iou_zero']) if metrics['iou_zero'] else 0,
            'num_samples': len(metrics['iou'])
        }
    
    # Insert summary at the beginning
    results.insert(0, summary)
    return results


def postprocess_prop_result(results):
    """
    Post-process proposal results.
    """
    if not results:
        return results
    
    # Calculate average metrics
    metrics_summary = {
        'num_samples': len(results)
    }
    
    results.insert(0, metrics_summary)
    return results


def save_results(results, save_path):
    """
    Save results to a JSON or JSONL file.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save the results
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if save_path.endswith(".json"):
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
    elif save_path.endswith(".jsonl"):
        with open(save_path, "w") as f:
            for info in results:
                f.write(json.dumps(info) + "\n")
    else:
        raise ValueError("Unsupported file format. Use .json or .jsonl")
    
    print(f"Results saved at: {save_path}")
    
    # Print summary if available
    if results and isinstance(results[0], dict) and 'overall' in results[0]:
        print("\n=== Evaluation Summary ===")
        summary = results[0]
        if 'overall' in summary:
            print(f"Overall Metrics:")
            for key, value in summary['overall'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        if 'by_type' in summary:
            print(f"\nMetrics by Type:")
            for task_type, metrics in summary['by_type'].items():
                print(f"  {task_type}:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.4f}")
                    else:
                        print(f"    {key}: {value}")
