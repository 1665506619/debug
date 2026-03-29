#!/usr/bin/env python3
"""Smoke-test the real video dataset interface used by SFTDataset + DataCollator.

This script intentionally avoids model forward/backward and only verifies that:
1. A video sample can pass through the real `SFTDataset.__getitem__`
2. Multiple samples can be merged by the real `DataCollator`

Supported modes:
- `--demo`: create minimal frame-list / frame-dir samples in a temp directory
- `--input-json <path>`: read a user-provided JSON/JSONL manifest
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _inject_package_shims(repo_root: Path) -> None:
    """Import local modules without executing heavyweight package __init__ files."""

    if "qwen_vl_utils" not in sys.modules:
        qwen_stub = types.ModuleType("qwen_vl_utils")

        def _missing_process_vision_info(*args, **kwargs):
            raise RuntimeError(
                "qwen_vl_utils.process_vision_info is unavailable in this environment. "
                "The dataset-debug script does not rely on it directly."
            )

        qwen_stub.process_vision_info = _missing_process_vision_info
        sys.modules["qwen_vl_utils"] = qwen_stub

    if "easy_vlm" not in sys.modules:
        easy_vlm_pkg = types.ModuleType("easy_vlm")
        easy_vlm_pkg.__path__ = [str(repo_root / "easy_vlm")]
        sys.modules["easy_vlm"] = easy_vlm_pkg

    if "easy_vlm.models" not in sys.modules:
        models_pkg = types.ModuleType("easy_vlm.models")
        models_pkg.__path__ = [str(repo_root / "easy_vlm" / "models")]
        sys.modules["easy_vlm.models"] = models_pkg

    if "easy_vlm.training" not in sys.modules:
        training_pkg = types.ModuleType("easy_vlm.training")
        training_pkg.__path__ = [str(repo_root / "easy_vlm" / "training")]
        sys.modules["easy_vlm.training"] = training_pkg

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _import_runtime_modules(repo_root: Path):
    _inject_package_shims(repo_root)

    try:
        import easy_vlm.models.qwen3_vl  # noqa: F401
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoProcessor,
            AutoTokenizer,
            AutoVideoProcessor,
        )
        from easy_vlm.constants import (
            REGION_TOKEN,
            REF_END_TOKEN,
            REF_START_TOKEN,
            SEG_END_TOKEN,
            SEG_START_TOKEN,
            SEG_TOKEN,
        )
        dataset_module = importlib.import_module("easy_vlm.training.dataset")
        training_utils = importlib.import_module("easy_vlm.training.utils")
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import required runtime dependencies. "
            "Please make sure the active Python environment has at least: "
            "`torch`, `transformers`, `datasets`, `pycocotools`, and repo-local imports available."
        ) from exc

    safe_rank0_print = lambda *args, **kwargs: print(*args, **kwargs)
    training_utils.rank0_print = safe_rank0_print
    dataset_module.rank0_print = safe_rank0_print

    return SimpleNamespace(
        AutoConfig=AutoConfig,
        AutoImageProcessor=AutoImageProcessor,
        AutoProcessor=AutoProcessor,
        AutoTokenizer=AutoTokenizer,
        AutoVideoProcessor=AutoVideoProcessor,
        REGION_TOKEN=REGION_TOKEN,
        SEG_TOKEN=SEG_TOKEN,
        REF_START_TOKEN=REF_START_TOKEN,
        REF_END_TOKEN=REF_END_TOKEN,
        SEG_START_TOKEN=SEG_START_TOKEN,
        SEG_END_TOKEN=SEG_END_TOKEN,
        SFTDataset=dataset_module.SFTDataset,
        DataCollator=dataset_module.DataCollator,
    )


@dataclass
class DebugSampleSpec:
    sample_type: str
    json_path: Path
    data_root: Path
    description: str


def _build_processors_and_config(args: argparse.Namespace, runtime) -> Tuple[Any, Any, Any]:
    config = runtime.AutoConfig.from_pretrained(args.model_path)
    tokenizer = runtime.AutoTokenizer.from_pretrained(args.model_path)
    image_processor = runtime.AutoImageProcessor.from_pretrained(args.model_path)
    video_processor = runtime.AutoVideoProcessor.from_pretrained(
        args.model_path,
        use_token_compression=args.use_token_compression,
    )
    processor = runtime.AutoProcessor.from_pretrained(
        args.model_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        video_processor=video_processor,
    )

    processor.tokenizer.add_tokens([runtime.REGION_TOKEN], special_tokens=True)
    processor.tokenizer.add_tokens(
        [
            runtime.SEG_TOKEN,
            runtime.REF_START_TOKEN,
            runtime.REF_END_TOKEN,
            runtime.SEG_START_TOKEN,
            runtime.SEG_END_TOKEN,
        ],
        special_tokens=True,
    )

    config.region_token_index = processor.tokenizer.convert_tokens_to_ids(runtime.REGION_TOKEN)
    config.seg_token_index = processor.tokenizer.convert_tokens_to_ids(runtime.SEG_TOKEN)
    config.seg_start_token_index = processor.tokenizer.convert_tokens_to_ids(runtime.SEG_START_TOKEN)
    config.seg_end_token_index = processor.tokenizer.convert_tokens_to_ids(runtime.SEG_END_TOKEN)
    config.ref_start_token_index = processor.tokenizer.convert_tokens_to_ids(runtime.REF_START_TOKEN)
    config.ref_end_token_index = processor.tokenizer.convert_tokens_to_ids(runtime.REF_END_TOKEN)

    config.max_seg_nums = args.max_seg_nums
    config.seg_encoder = args.seg_encoder
    config.seg_decoder = args.seg_decoder
    config.mask_decoder_model = args.mask_decoder_model
    config.dice_loss_weight = 0.5
    config.bce_loss_weight = 2.0
    config.cls_loss_weight = 1.0
    config.loss_sample_points = args.loss_sample_points

    seg_processor = runtime.AutoProcessor.from_pretrained(args.mask_decoder_model)
    return config, processor, seg_processor


def _make_data_args(json_path: Path, data_root: Path, output_dir: Path, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        ann_path=[json_path.name],
        data_root=str(data_root),
        data_path_root=str(json_path.parent),
        data_cache_dir=str(output_dir / "hf_cache"),
        max_seg_nums=args.max_seg_nums,
        skip_none=args.skip_none,
        output_dir=str(output_dir),
    )


def _build_dataset(
    json_path: Path,
    data_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    runtime,
    model_config,
    processor,
    seg_processor,
):
    data_args = _make_data_args(json_path=json_path, data_root=data_root, output_dir=output_dir, args=args)
    dataset = runtime.SFTDataset(
        model_config=model_config,
        processor=processor,
        seg_processor=seg_processor,
        model_max_length=args.model_max_length,
        mm_max_length=args.mm_max_length,
        fps=args.fps,
        max_frames=args.max_frames,
        dataloader_num_workers=0,
        data_args=data_args,
        requires_length=False,
        use_multi_objs=args.use_multi_objs,
    )
    return dataset


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"Unsupported JSON payload in {path}")


def _infer_sample_type(record: Dict[str, Any]) -> str:
    video = record.get("video")
    if isinstance(video, list):
        return "frame_list"
    if isinstance(video, str):
        lower = video.lower()
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".gif", ".webm")):
            return "mp4"
        return "frame_dir"
    return "unknown"


def _shape_of(value: Any) -> Optional[Tuple[int, ...]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(x) for x in shape)
    except TypeError:
        return tuple(shape)


def _format_grid(value: Any) -> str:
    if value is None:
        return "None"
    if hasattr(value, "detach"):
        return str(value.detach().cpu().tolist())
    return str(value)


def _describe_batch_list(values: Sequence[Any]) -> str:
    if values is None:
        return "None"
    parts = []
    for idx, value in enumerate(values):
        parts.append(f"{idx}:{type(value).__name__}:{_shape_of(value)}")
    return "[" + ", ".join(parts) + "]"


def _print_sample_summary(sample_type: str, sample: Dict[str, Any]) -> None:
    pixel_values_videos = sample.get("pixel_values_videos")
    print(f"\n=== Dataset Sample: {sample_type} ===")
    print(f"input_ids.shape: {_shape_of(sample.get('input_ids'))}")
    print(f"has pixel_values_videos: {pixel_values_videos is not None}")
    print(f"pixel_values_videos.shape: {_shape_of(pixel_values_videos)}")
    print(f"video_grid_thw: {_format_grid(sample.get('video_grid_thw'))}")
    print(f"sam_images.shape: {_shape_of(sample.get('sam_images'))}")
    masks = sample.get("masks")
    print(f"masks: type={type(masks).__name__}, shape={_shape_of(masks)}")
    print(f"mask_ids: {sample.get('mask_ids')}")
    print(f"mask_valid: {sample.get('mask_valid')}")
    print(f"mask_type: {sample.get('mask_type')}")


def _print_batch_summary(batch: Dict[str, Any], sequence_packing: bool) -> None:
    print("\n=== Collated Batch ===")
    print(f"input_ids.shape: {_shape_of(batch.get('input_ids'))}")
    if sequence_packing:
        print("attention_mask.shape: N/A (sequence packing path does not return attention_mask)")
    else:
        print(f"attention_mask.shape: {_shape_of(batch.get('attention_mask'))}")
    print(f"pixel_values_videos.shape: {_shape_of(batch.get('pixel_values_videos'))}")
    print(f"video_grid_thw.shape: {_shape_of(batch.get('video_grid_thw'))}")
    print(f"sam_images batch structure: {_describe_batch_list(batch.get('sam_images', []))}")
    print(f"masks batch structure: {_describe_batch_list(batch.get('masks', []))}")
    print(f"mask_ids: {batch.get('mask_ids')}")


def _run_single_sample(
    dataset,
    raw_record: Dict[str, Any],
    index: int,
    sample_type: str,
    precheck_preprocess: bool = False,
):
    if raw_record is not None and precheck_preprocess:
        try:
            dataset._preprocess(dataset._dataset[index])
        except Exception as exc:
            print(f"\n=== Dataset Sample: {sample_type} ===")
            print(f"preprocess error: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return None

    try:
        sample = dataset[index]
    except Exception as exc:
        print(f"\n=== Dataset Sample: {sample_type} ===")
        print(f"getitem error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None

    requested = index
    returned = sample.get("data_index")
    if returned != requested:
        print(
            f"\n=== Dataset Sample: {sample_type} ===\n"
            f"warning: requested index {requested}, but `SFTDataset.__getitem__` returned backup index {returned}"
        )

    _print_sample_summary(sample_type=sample_type, sample=sample)
    return sample


def _encode_binary_mask(mask: "np.ndarray") -> Dict[str, Any]:
    from pycocotools import mask as mask_utils
    import numpy as np

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": list(encoded["size"]), "counts": counts}


def _build_demo_record(
    *,
    video_value: Any,
    height: int,
    width: int,
    mask_by_key: Dict[str, Dict[str, Any]],
    frame_idx: Optional[List[int]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "type": "grounding",
        "video": video_value,
        "height": height,
        "width": width,
        "annotation": [
            {
                "type": "phrase",
                "text": "green square",
                "ann_type": "mask",
                "ann": [mask_by_key],
            }
        ],
    }
    if frame_idx is not None:
        record["frame_idx"] = frame_idx
    return record


def _create_demo_assets(root: Path) -> List[DebugSampleSpec]:
    import numpy as np
    from PIL import Image

    samples_dir = root / "assets"
    samples_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = samples_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    width = 64
    height = 64
    frame_paths: List[Path] = []
    masks_by_stem: Dict[str, Dict[str, Any]] = {}
    masks_by_index: Dict[str, Dict[str, Any]] = {}
    masks_by_float_index: Dict[str, Dict[str, Any]] = {}

    for idx in range(4):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = 20
        frame[..., 1] = 20
        frame[..., 2] = 20

        x0 = 2 + idx * 4
        y0 = 4 + idx
        x1 = min(x0 + 8, width)
        y1 = min(y0 + 6, height)
        frame[y0:y1, x0:x1, 1] = 220
        frame[y0:y1, x0:x1, 0] = 40
        frame[y0:y1, x0:x1, 2] = 40

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        rle = _encode_binary_mask(mask)

        stem = f"{idx:05d}"
        frame_path = frame_dir / f"{stem}.png"
        Image.fromarray(frame).save(frame_path)
        frame_paths.append(frame_path)
        masks_by_stem[stem] = rle
        masks_by_index[str(idx)] = rle
        masks_by_float_index[str(float(idx))] = rle

    all_mask_keys = dict(masks_by_stem)
    all_mask_keys.update(masks_by_index)
    all_mask_keys.update(masks_by_float_index)

    specs: List[DebugSampleSpec] = []

    frame_list_record = _build_demo_record(
        video_value=[str(path.relative_to(root)) for path in frame_paths],
        height=height,
        width=width,
        mask_by_key=all_mask_keys,
    )
    frame_list_json = root / "frame_list.jsonl"
    frame_list_json.write_text(json.dumps(frame_list_record, ensure_ascii=False) + "\n", encoding="utf-8")
    specs.append(
        DebugSampleSpec(
            sample_type="frame_list",
            json_path=frame_list_json,
            data_root=root,
            description="video is a list[path] frame sample",
        )
    )

    frame_dir_record = _build_demo_record(
        video_value=str(frame_dir.relative_to(root)),
        height=height,
        width=width,
        mask_by_key=all_mask_keys,
        frame_idx=[1, 3],
    )
    frame_dir_json = root / "frame_dir.jsonl"
    frame_dir_json.write_text(json.dumps(frame_dir_record, ensure_ascii=False) + "\n", encoding="utf-8")
    specs.append(
        DebugSampleSpec(
            sample_type="frame_dir",
            json_path=frame_dir_json,
            data_root=root,
            description="video is a frame directory plus frame_idx sample",
        )
    )

    mp4_error = None
    mp4_path = root / "demo.mp4"
    try:
        frames_rgb = [np.array(Image.open(path).convert("RGB")) for path in frame_paths]
        try:
            import imageio.v2 as imageio

            imageio.mimwrite(mp4_path, frames_rgb, fps=2, macro_block_size=None)
        except Exception:
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(mp4_path), fourcc, 2.0, (width, height))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open the target mp4 file")
            for frame in frames_rgb:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        if not mp4_path.exists():
            raise RuntimeError("mp4 file writer returned without creating the file")

        mp4_record = _build_demo_record(
            video_value=str(mp4_path.relative_to(root)),
            height=height,
            width=width,
            mask_by_key=all_mask_keys,
        )
        mp4_json = root / "mp4.jsonl"
        mp4_json.write_text(json.dumps(mp4_record, ensure_ascii=False) + "\n", encoding="utf-8")
        specs.append(
            DebugSampleSpec(
                sample_type="mp4",
                json_path=mp4_json,
                data_root=root,
                description="video is an mp4 sample",
            )
        )
    except Exception as exc:
        mp4_error = exc

    print(f"demo root: {root}")
    print("demo samples:")
    for spec in specs:
        print(f"- {spec.sample_type}: {spec.description} -> {spec.json_path}")
    if mp4_error is not None:
        print(f"- mp4: skipped mp4 smoke test ({type(mp4_error).__name__}: {mp4_error})")

    return specs


def _build_input_json_spec(args: argparse.Namespace) -> List[DebugSampleSpec]:
    input_path = Path(args.input_json).resolve()
    records = _load_json_records(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")
    sample_type = _infer_sample_type(records[min(args.sample_index, len(records) - 1)])
    data_root = Path(args.data_root).resolve() if args.data_root else input_path.parent
    return [
        DebugSampleSpec(
            sample_type=sample_type,
            json_path=input_path,
            data_root=data_root,
            description="user-provided manifest",
        )
    ]


def _pick_collator_samples(
    successful_samples: List[Tuple[str, Dict[str, Any]]],
    selected_type: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    if len(successful_samples) >= 2:
        return successful_samples[:2]
    if len(successful_samples) == 1:
        return [successful_samples[0], successful_samples[0]]
    return []


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug the real video dataset interface used by SFTDataset.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json", type=str, help="Path to a JSON/JSONL manifest to debug.")
    source.add_argument("--demo", action="store_true", help="Create and debug minimal demo video samples.")

    parser.add_argument("--model-path", type=str, required=True, help="Model path used for AutoConfig/AutoProcessor.")
    parser.add_argument(
        "--mask-decoder-model",
        type=str,
        required=True,
        help="Segmentation processor path used for SAM image preprocessing.",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Optional root for media paths in --input-json mode.")
    parser.add_argument("--fps", type=int, default=1, help="Video sampling fps passed to SFTDataset.")
    parser.add_argument("--max-frames", type=int, default=8, help="Max frames passed to SFTDataset.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index for --input-json mode.")
    parser.add_argument("--sequence-packing", action="store_true", help="Use DataCollator(sequence_packing=True).")
    parser.add_argument("--model-max-length", type=int, default=16384)
    parser.add_argument("--mm-max-length", type=int, default=10240)
    parser.add_argument("--max-seg-nums", type=int, default=10)
    parser.add_argument("--seg-encoder", type=str, default="sam3")
    parser.add_argument("--seg-decoder", type=str, default="sam3")
    parser.add_argument("--loss-sample-points", action="store_true")
    parser.add_argument(
        "--skip-none",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to skip annotations whose mask is None. Use --no-skip-none to keep them.",
    )
    parser.add_argument("--use-multi-objs", action="store_true", default=False)
    parser.add_argument("--use-token-compression", action="store_true", default=False)
    parser.add_argument(
        "--precheck-preprocess",
        action="store_true",
        help="Optionally call dataset._preprocess(...) before dataset[idx] to surface the original exception.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    runtime = _import_runtime_modules(repo_root)

    print("initializing processors/config...")
    model_config, processor, seg_processor = _build_processors_and_config(args=args, runtime=runtime)

    if args.demo:
        debug_root = Path(tempfile.mkdtemp(prefix="debug_video_dataset_", dir=tempfile.gettempdir()))
        sample_specs = _create_demo_assets(debug_root)
        output_dir = debug_root / "outputs"
    else:
        sample_specs = _build_input_json_spec(args)
        output_dir = Path(tempfile.mkdtemp(prefix="debug_video_dataset_outputs_", dir=tempfile.gettempdir()))

    output_dir.mkdir(parents=True, exist_ok=True)
    data_collator = runtime.DataCollator(
        processor=processor,
        sequence_packing=args.sequence_packing,
    )

    successful_samples: List[Tuple[str, Dict[str, Any]]] = []
    user_input_records: Optional[List[Dict[str, Any]]] = None
    if args.input_json:
        user_input_records = _load_json_records(Path(args.input_json))

    for spec in sample_specs:
        dataset = _build_dataset(
            json_path=spec.json_path,
            data_root=spec.data_root,
            output_dir=output_dir,
            args=args,
            runtime=runtime,
            model_config=model_config,
            processor=processor,
            seg_processor=seg_processor,
        )

        if args.input_json and user_input_records is not None:
            if args.sample_index >= len(user_input_records):
                raise IndexError(
                    f"sample-index {args.sample_index} is out of range for {len(user_input_records)} records in {args.input_json}"
                )
            raw_record = user_input_records[args.sample_index]
            sample = _run_single_sample(
                dataset=dataset,
                raw_record=raw_record,
                index=args.sample_index,
                sample_type=spec.sample_type,
                precheck_preprocess=args.precheck_preprocess,
            )
        else:
            raw_record = _load_json_records(spec.json_path)[0]
            sample = _run_single_sample(
                dataset=dataset,
                raw_record=raw_record,
                index=0,
                sample_type=spec.sample_type,
                precheck_preprocess=args.precheck_preprocess,
            )

        if sample is not None:
            successful_samples.append((spec.sample_type, sample))

    collator_inputs = _pick_collator_samples(successful_samples)
    if len(collator_inputs) < 2:
        print("\ncollator skipped: fewer than two successful samples were produced.")
        return 0

    collator_types = [sample_type for sample_type, _ in collator_inputs]
    collator_payload = [sample for _, sample in collator_inputs]
    if len(collator_inputs) == 2 and collator_types[0] == collator_types[1] and len(successful_samples) == 1:
        print(f"\ncollator input: duplicating the only successful sample ({collator_types[0]}) to exercise batch merge.")
    else:
        print(f"\ncollator input sample types: {collator_types}")

    try:
        batch = data_collator(collator_payload)
    except Exception as exc:
        print(f"\ncollator error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    _print_batch_summary(batch=batch, sequence_packing=args.sequence_packing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
