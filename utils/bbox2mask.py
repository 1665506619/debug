import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Processor, Sam2Model
from pycocotools import mask as maskUtils


# -------------------------
# Utils
# -------------------------
def atomic_write_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def load_json_any(path: str):
    # 支持 .json (list/dict) 或 .jsonl
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        records = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    else:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

def xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]

def clip_xyxy(box_xyxy, W, H):
    x0, y0, x1, y1 = box_xyxy
    x0 = max(0.0, min(float(x0), float(W - 1)))
    y0 = max(0.0, min(float(y0), float(H - 1)))
    x1 = max(0.0, min(float(x1), float(W)))
    y1 = max(0.0, min(float(y1), float(H)))
    if x1 <= x0:
        x1 = min(float(W), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(H), y0 + 1.0)
    return [x0, y0, x1, y1]

def ensure_list_of_boxes(bbox_field):
    """
    兼容：
      bbox: [x,y,w,h]
      bbox: [[x,y,w,h], ...]
    """
    if bbox_field is None:
        return []
    if isinstance(bbox_field, (list, tuple)) and len(bbox_field) == 4 and isinstance(bbox_field[0], (int, float)):
        return [bbox_field]
    return bbox_field

def singleMask2rle(mask_2d):
    """
    mask_2d: (H,W) bool/0-1/uint8
    return: {"size":[H,W], "counts": "..."}  (COCO RLE)
    """
    if mask_2d is None:
        return None
    m = np.asarray(mask_2d)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W), got {m.shape}")
    m = (m > 0).astype(np.uint8)
    rle = maskUtils.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

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

@torch.inference_mode()
def process_one_record(record, image_root: str, model, processor, device, thr=0.0):
    """
    对单条 record：把所有 bbox->mask->rle 写到 annotation[*]["mask_rles"]
    """
    img_path = Path(image_root) / record["image"]
    raw_image = Image.open(img_path).convert("RGB")
    W, H = raw_image.size

    flat_boxes_xyxy = []
    index_map = []  # (ann_idx, local_box_idx)

    anns = record.get("annotation", [])
    for ann_idx, ann in enumerate(anns):
        boxes_xywh = ensure_list_of_boxes(ann.get("bbox"))
        if not boxes_xywh:
            continue
        for b_i, b_xywh in enumerate(boxes_xywh):
            b_xyxy = clip_xyxy(xywh_to_xyxy(b_xywh), W, H)
            flat_boxes_xyxy.append(b_xyxy)
            index_map.append((ann_idx, b_i))

    # 先统一补齐 mask_rles 槽位（保持和 bbox 数量一致）
    for ann in anns:
        boxes_xywh = ensure_list_of_boxes(ann.get("bbox"))
        if boxes_xywh:
            ann["mask"] = [None] * len(boxes_xywh)

    if not flat_boxes_xyxy:
        return record

    # batch=1, num_boxes=N
    input_boxes = [flat_boxes_xyxy]
    inputs = processor(images=raw_image, input_boxes=input_boxes, return_tensors="pt").to(device)

    use_amp = (device.type == "cuda")
    if use_amp:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs, multimask_output=False)
    else:
        outputs = model(**inputs, multimask_output=False)

    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    # (num_boxes, 1, H, W) -> (num_boxes, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    masks_np = masks.numpy()
    masks_bin = masks_np 

    for i, (ann_idx, local_i) in enumerate(index_map):
        record["annotation"][ann_idx]["mask"][local_i] = singleMask2rle(masks_bin[i])

    return record


def setup_distributed():
    """
    兼容 torchrun（多进程）和普通 python（单进程）
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default='/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/chatrex.json')
    parser.add_argument("--image_root", default='/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data')
    parser.add_argument("--out_dir", default='/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/chatrex_mask')
    parser.add_argument("--model_name", default="facebook/sam2-hiera-large")
    parser.add_argument("--save_every", type=int, default=1, help="每多少条保存一次")
    parser.add_argument("--thr", type=float, default=0.0, help="mask 二值化阈值")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    # 设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 每卡独立输出和状态文件
    out_jsonl = out_dir / f"shard_{rank:02d}.jsonl"
    state_path = out_dir / f"shard_{rank:02d}.state.json"

    # 读全量数据（简单可靠；如果数据超大再做流式）
    records = load_json_any(args.json_path)
    if isinstance(records, dict):
        records = [records]
    N = len(records)

    # 连续分片：rank 0..world_size-1
    shard_size = math.ceil(N / world_size)
    shard_start = rank * shard_size
    shard_end = min(N, shard_start + shard_size)
    shard_len = max(0, shard_end - shard_start)

    if shard_len == 0:
        if rank == 0:
            print(f"[rank {rank}] Nothing to do: N={N}, world_size={world_size}")
        return

    # 断点续跑：读取 state.json 里的 next_pos（在本 shard 内的偏移）
    next_pos = 0
    if state_path.exists():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
            next_pos = int(st.get("next_pos", 0))
            next_pos = max(0, min(next_pos, shard_len))
        except Exception:
            next_pos = 0

    # 加载模型（每个进程/每张卡各加载一份）
    model = Sam2Model.from_pretrained(args.model_name).to(device).eval()
    processor = Sam2Processor.from_pretrained(args.model_name)

    # 处理循环：每 save_every 条落盘一次
    buffer = []
    processed = 0

    # 如果是续跑，提示一下
    if next_pos > 0:
        print(f"[rank {rank}] Resume from shard_pos={next_pos}/{shard_len} (global_idx={shard_start + next_pos})")

    for pos in range(next_pos, shard_len):
        global_idx = shard_start + pos
        rec = records[global_idx]

        # 可选：写入索引，方便后面 merge 回原顺序
        rec["_global_idx"] = global_idx

        try:
            rec = process_one_record(
                rec, image_root=args.image_root,
                model=model, processor=processor,
                device=device, thr=args.thr
            )
            rec["_ok"] = True
        except Exception as e:
            # 不中断整体：记录错误，方便排查
            rec["_ok"] = False
            rec["_error"] = repr(e)

        buffer.append(rec)
        processed += 1

        # 每 save_every 条保存一次
        if len(buffer) >= args.save_every:
            with out_jsonl.open("a", encoding="utf-8") as f:
                for r in buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            buffer.clear()

            # 更新 state：下一条位置
            atomic_write_json(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "shard_start": shard_start,
                    "shard_end": shard_end,
                    "next_pos": pos + 1,
                    "done": False,
                },
                state_path
            )
            print(f"[rank {rank}] Saved {args.save_every} records. progress={pos+1}/{shard_len}")

    # 收尾：把剩余 buffer 落盘
    if buffer:
        with out_jsonl.open("a", encoding="utf-8") as f:
            for r in buffer:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        buffer.clear()

    # 标记 done
    atomic_write_json(
        {
            "rank": rank,
            "world_size": world_size,
            "shard_start": shard_start,
            "shard_end": shard_end,
            "next_pos": shard_len,
            "done": True,
        },
        state_path
    )
    print(f"[rank {rank}] Done. shard [{shard_start}, {shard_end}) processed.")

    # 结束分布式
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
