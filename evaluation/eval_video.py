import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.append("./")

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from easy_vlm.models import VideoSegEngine, load_pretrained_model
from easy_vlm.models.sam3_full.sam3_video_predictor import Sam3VideoPredictor
from evaluation.metrics import db_eval_boundary, db_eval_iou


def single_mask_to_rle(mask):
    if mask is None:
        return None
    rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def ann_to_mask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = mask_utils.frPyObjects(mask_ann, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(mask_ann["counts"], list):
        rle = mask_utils.frPyObjects(mask_ann, h, w)
    else:
        rle = mask_ann
    return mask_utils.decode(rle)


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_pil_frames(frame_paths):
    return [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]


def build_conversation(video_content, instruction):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_content},
                {"type": "text", "text": instruction},
            ],
        }
    ]


def infer_phrase_from_instruction(instruction):
    if "'" in instruction:
        parts = instruction.split("'")
        if len(parts) >= 3 and parts[1].strip():
            return parts[1].strip()
    return None


def aggregate_video_results(video_results):
    if not video_results:
        return None

    num_frames = max(result["num_frames"] for result in video_results)
    merged_masks = []
    for frame_idx in range(num_frames):
        frame_mask_union = None
        for result in video_results:
            frame_masks = result["masks"][frame_idx]
            if frame_masks is None:
                continue
            frame_masks = np.asarray(frame_masks).astype(bool)
            if frame_masks.ndim == 2:
                frame_masks = frame_masks[None, ...]
            frame_union = np.any(frame_masks, axis=0)
            frame_mask_union = (
                frame_union
                if frame_mask_union is None
                else np.logical_or(frame_mask_union, frame_union)
            )
        if frame_mask_union is None:
            merged_masks.append(None)
        else:
            merged_masks.append(frame_mask_union.astype(np.uint8))
    return merged_masks


def resize_pred_masks_if_needed(pred_masks, height, width):
    if pred_masks is None:
        return None
    if pred_masks.shape[-2:] == (height, width):
        return pred_masks
    pred_masks_t = torch.from_numpy(pred_masks[:, None].astype(np.float32))
    pred_masks_t = torch.nn.functional.interpolate(
        pred_masks_t,
        size=(height, width),
        mode="nearest",
    )
    return pred_masks_t[:, 0].numpy().astype(bool)


class VideoEvalDataset(Dataset):
    def __init__(self, video_folder, data_list):
        data_list_new = []
        for data in data_list:
            video_rel_path = data["video"]
            video_abs_path = (
                video_rel_path
                if os.path.isabs(video_rel_path)
                else os.path.join(video_folder, video_rel_path)
            )

            if "Expression" in data or "expression" in data:
                expression = data.get("Expression", data.get("expression"))
                if os.path.isdir(video_abs_path) and len(os.listdir(video_abs_path)) == 0:
                    print(f"Video path does not exist or is empty: {video_abs_path}")
                    continue
                instruction = f"Can you segment '{expression.rstrip('.').lower()}' in the video?"
                phrase = expression.rstrip(".")
            else:
                instruction = data["question"]
                phrase = infer_phrase_from_instruction(instruction)

            data_list_new.append(
                {
                    "video_path": video_abs_path,
                    "instruction": instruction,
                    "phrase": phrase,
                    "gt_mask": data["masks"],
                    "frame_names": data.get("frame_names"),
                }
            )
        self.data_list = data_list_new

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        height, width = None, None
        for mask_ann in data["gt_mask"]:
            if mask_ann is not None:
                height, width = mask_ann["size"]
                break
        if height is None:
            if data["frame_names"] is not None:
                first_frame_path = os.path.join(
                    data["video_path"], f"{data['frame_names'][0]}.jpg"
                )
                img = cv2.imread(first_frame_path)
                height, width = img.shape[:2]
            elif os.path.isdir(data["video_path"]):
                frame_files = sorted(
                    [
                        os.path.join(data["video_path"], x)
                        for x in os.listdir(data["video_path"])
                        if x.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
                img = cv2.imread(frame_files[0])
                height, width = img.shape[:2]
            else:
                cap = cv2.VideoCapture(data["video_path"])
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap.release()

        gt_masks = []
        for mask_ann in data["gt_mask"]:
            if mask_ann is not None:
                gt_masks.append(ann_to_mask(mask_ann))
            else:
                gt_masks.append(np.zeros((height, width), dtype=np.uint8))

        if data["frame_names"] is not None:
            frame_paths = [
                os.path.join(data["video_path"], f"{frame_name}.jpg")
                for frame_name in data["frame_names"]
            ]
            video_content = frame_paths
            video_resource = load_pil_frames(frame_paths)
        else:
            video_content = data["video_path"]
            video_resource = data["video_path"]

        return {
            "idx": idx,
            "instruction": data["instruction"],
            "phrase": data["phrase"],
            "gt_masks": gt_masks,
            "video_content": video_content,
            "video_resource": video_resource,
            "video_path": data["video_path"],
        }


def collate_fn(batch):
    return batch


def build_eval_dataloader(args, distributed):
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = VideoEvalDataset(args.video_folder, questions)
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset)
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
    )


def save_results(result, save_path):
    if len(result) == 0:
        print("Warning: No results to save!")
        return

    j = sum(item["j"] for item in result) / len(result)
    f = sum(item["f"] for item in result) / len(result)
    metrics = {"j": j, "f": f, "j&f": (j + f) / 2}
    result.insert(0, metrics)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith(".json"):
        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)
    elif save_path.endswith(".jsonl"):
        with open(save_path, "w") as f:
            for info in result:
                f.write(json.dumps(info) + "\n")
    else:
        raise ValueError("Unsupported file format.")
    print(f"Answer saved at:{save_path}")


def init_distributed():
    distributed = int(os.getenv("WORLD_SIZE", "1")) > 1
    if not distributed:
        return False, 0, 0

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    return True, local_rank, global_rank


def load_video_inference_stack(args, local_rank):
    device_map = {"": f"cuda:{local_rank}"} if torch.cuda.is_available() else "auto"
    tokenizer, model, processor = load_pretrained_model(
        args.model_path,
        None,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    if args.use_bfloat16 and torch.cuda.is_available():
        model.to(torch.bfloat16)

    predictor = Sam3VideoPredictor(
        checkpoint_path=args.sam3_video_checkpoint,
        bpe_path=args.sam3_bpe_path,
        async_loading_frames=args.async_loading_frames,
        video_loader_type=args.video_loader_type,
        compile=args.sam3_compile,
    )
    video_seg_engine = VideoSegEngine(predictor.model)
    model.set_video_seg_engine(video_seg_engine)
    return tokenizer, model, processor, video_seg_engine


def run_single_inference(sample, processor, tokenizer, model, video_seg_engine, args):
    conversation = build_conversation(
        video_content=sample["video_content"],
        instruction=sample["instruction"],
    )
    inputs = processor.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=args.fps,
        max_frames=args.max_frames,
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_output_ids, video_results = model.inference_video(
            video_resource_path=sample["video_resource"],
            tokenizer=tokenizer,
            phrases=[sample["phrase"]] if sample["phrase"] else None,
            start_frame=args.start_frame,
            video_seg_engine=video_seg_engine,
            video_init_kwargs={
                "async_loading_frames": args.async_loading_frames,
                "video_loader_type": args.video_loader_type,
            },
            max_frame_num_to_track=args.max_frame_num_to_track,
            propagate_both_directions=args.propagate_both_directions,
            **inputs,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

    output_text = processor.tokenizer.batch_decode(
        generated_output_ids, skip_special_tokens=False
    )[0].strip()
    pred_masks = aggregate_video_results(video_results)
    return output_text, pred_masks, video_results


def run_inference(args):
    distributed, local_rank, global_rank = init_distributed()
    tokenizer, model, processor, video_seg_engine = load_video_inference_stack(
        args, local_rank
    )
    val_loader = build_eval_dataloader(args, distributed)

    results = []
    for batch in tqdm(
        val_loader,
        desc=f"Rank {global_rank}",
        total=len(val_loader),
        position=local_rank,
    ):
        for sample in batch:
            output_text, pred_masks, video_results = run_single_inference(
                sample, processor, tokenizer, model, video_seg_engine, args
            )

            gt_masks = np.asarray(sample["gt_masks"]).astype(bool)
            h, w = gt_masks.shape[1], gt_masks.shape[2]
            if pred_masks is None:
                pred_masks = np.zeros((len(gt_masks), h, w), dtype=bool)
            else:
                pred_masks = [
                    np.zeros((h, w), dtype=bool) if mask is None else mask.astype(bool)
                    for mask in pred_masks
                ]
                pred_masks = np.asarray(pred_masks)
                pred_masks = resize_pred_masks_if_needed(pred_masks, h, w)

            mask_rles = [single_mask_to_rle(pred_mask.astype(np.uint8)) for pred_mask in pred_masks]

            if gt_masks.shape[0] == pred_masks.shape[0]:
                j = db_eval_iou(gt_masks, pred_masks).mean()
                f = db_eval_boundary(gt_masks, pred_masks).mean()
            else:
                print(
                    f"Data {sample['idx']}: GT frames ({gt_masks.shape[0]}) != Pred frames ({pred_masks.shape[0]}), "
                    "skip intermediate eval"
                )
                j = 0.0
                f = 0.0

            results.append(
                {
                    "idx": sample["idx"],
                    "instruction": sample["instruction"],
                    "prediction": output_text,
                    "j": float(j),
                    "f": float(f),
                    "mask_rle": mask_rles,
                    "video_path": sample["video_path"],
                    "num_segments": 0 if video_results is None else len(video_results),
                }
            )

    if distributed:
        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.gather_object(
            obj=results,
            object_gather_list=gathered_results if global_rank == 0 else None,
            dst=0,
        )
        if global_rank == 0:
            results = sum(gathered_results, [])
            save_results(results, args.output_file)
        dist.destroy_process_group()
    else:
        save_results(results, args.output_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sam3_video_checkpoint", type=str, required=True)
    parser.add_argument("--sam3_bpe_path", type=str, default=None)
    parser.add_argument("--video_folder", type=str, default="/mnt")
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frame-num-to-track", type=int, default=None)
    parser.add_argument(
        "--propagate-both-directions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--async-loading-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--video-loader-type",
        type=str,
        default="cv2",
        choices=["cv2", "torchcodec"],
    )
    parser.add_argument(
        "--sam3-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument(
        "--use-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
