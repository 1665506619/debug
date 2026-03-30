#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_video_meta_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_videos.json")


def _has_valid_mask(exp_meta: Dict[str, Any], mask_dict: Dict[str, Any]) -> bool:
    for anno_id in exp_meta["anno_id"]:
        anno_masks = mask_dict.get(str(anno_id))
        if anno_masks is None:
            raise KeyError(f"Missing anno_id={anno_id} in mask_dict")
        if any(mask is not None for mask in anno_masks):
            return True
    return False


def _build_expression_sample(
    video_rel_path: str,
    video_meta: Dict[str, Any],
    exp_id: str,
    exp_meta: Dict[str, Any],
    video_meta_output_path: Path,
    mask_dict_path: Path,
) -> Dict[str, Any]:
    return {
        "type": "grounding",
        "source": "revos_train",
        "source_dataset": "revos",
        "source_split": "train",
        "video_id": video_rel_path,
        "video_rel_path": video_rel_path,
        "exp_id": str(exp_id),
        "expression": exp_meta["exp"],
        "anno_ids": list(exp_meta["anno_id"]),
        "obj_ids": list(exp_meta.get("obj_id", [])),
        "height": video_meta["height"],
        "width": video_meta["width"],
        "type_id": exp_meta.get("type_id"),
        "revos_video_meta_file": video_meta_output_path.name,
        "revos_mask_dict_path": str(mask_dict_path),
    }


def construct_revos_train(
    meta_path: Path,
    mask_dict_path: Path,
    data_root: Path,
    output_path: Path,
    video_meta_output_path: Optional[Path] = None,
    max_samples: Optional[int] = None,
    max_videos: Optional[int] = None,
    skip_missing_videos: bool = True,
) -> Tuple[Dict[str, int], Path]:
    meta = _load_json(meta_path)
    mask_dict = _load_json(mask_dict_path)

    if video_meta_output_path is None:
        video_meta_output_path = _default_video_meta_output_path(output_path)

    output_samples: List[Dict[str, Any]] = []
    video_sidecar: Dict[str, Dict[str, Any]] = {}
    stats = {
        "videos_seen": 0,
        "videos_missing": 0,
        "videos_kept": 0,
        "expressions_seen": 0,
        "expressions_kept": 0,
        "expressions_dropped_empty_mask": 0,
    }

    for video_rel_path, video_meta in meta["videos"].items():
        stats["videos_seen"] += 1
        if max_videos is not None and stats["videos_kept"] >= max_videos:
            break

        video_dir = data_root / video_rel_path
        if not video_dir.is_dir():
            stats["videos_missing"] += 1
            if skip_missing_videos:
                continue
            raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

        kept_any_expression = False
        for exp_id, exp_meta in video_meta["expressions"].items():
            stats["expressions_seen"] += 1
            if not _has_valid_mask(exp_meta, mask_dict):
                stats["expressions_dropped_empty_mask"] += 1
                continue

            output_samples.append(
                _build_expression_sample(
                    video_rel_path=video_rel_path,
                    video_meta=video_meta,
                    exp_id=exp_id,
                    exp_meta=exp_meta,
                    video_meta_output_path=video_meta_output_path,
                    mask_dict_path=mask_dict_path,
                )
            )
            stats["expressions_kept"] += 1
            kept_any_expression = True

            if max_samples is not None and stats["expressions_kept"] >= max_samples:
                break

        if kept_any_expression:
            video_sidecar[video_rel_path] = {
                "video_rel_path": video_rel_path,
                "frame_names": list(video_meta["frames"]),
                "height": video_meta["height"],
                "width": video_meta["width"],
            }
            stats["videos_kept"] += 1

        if max_samples is not None and stats["expressions_kept"] >= max_samples:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_meta_output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_samples, f, ensure_ascii=False, indent=2)

    with video_meta_output_path.open("w", encoding="utf-8") as f:
        json.dump({"videos": video_sidecar}, f, ensure_ascii=False, indent=2)

    return stats, video_meta_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct REVoS train annotations into a compact SFTDataset video schema.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("/root/InstructSAM/video_dataset/revos/meta_expressions_train_.json"),
    )
    parser.add_argument(
        "--mask-dict-path",
        type=Path,
        default=Path("/root/InstructSAM/video_dataset/revos/mask_dict.json"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/InstructSAM/video_dataset/revos"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/root/InstructSAM/outputs/revos_train/revos_train_sft.json"),
    )
    parser.add_argument(
        "--video-meta-output-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument(
        "--skip-missing-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats, video_meta_output_path = construct_revos_train(
        meta_path=args.meta_path,
        mask_dict_path=args.mask_dict_path,
        data_root=args.data_root,
        output_path=args.output_path,
        video_meta_output_path=args.video_meta_output_path,
        max_samples=args.max_samples,
        max_videos=args.max_videos,
        skip_missing_videos=args.skip_missing_videos,
    )

    print(f"Saved REVoS train SFT data to: {args.output_path}")
    print(f"Saved REVoS train video metadata to: {video_meta_output_path}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
