#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
import traceback
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _inject_package_shims(repo_root: Path) -> None:
    debug_root = repo_root / "debug"

    if "qwen_vl_utils" not in sys.modules:
        qwen_stub = types.ModuleType("qwen_vl_utils")

        def _missing_process_vision_info(*args, **kwargs):
            raise RuntimeError("qwen_vl_utils.process_vision_info is unavailable in this environment.")

        qwen_stub.process_vision_info = _missing_process_vision_info
        sys.modules["qwen_vl_utils"] = qwen_stub

    if "easy_vlm" not in sys.modules:
        easy_vlm_pkg = types.ModuleType("easy_vlm")
        easy_vlm_pkg.__path__ = [str(debug_root / "easy_vlm")]
        sys.modules["easy_vlm"] = easy_vlm_pkg

    for path in (repo_root, debug_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_runtime(repo_root: Path):
    _inject_package_shims(repo_root)

    trainer_module = importlib.import_module("easy_vlm.training.trainer")
    training_utils = importlib.import_module("easy_vlm.training.utils")
    construct_module = importlib.import_module("debug.evaluation.data_construction.construct_revos_train")

    safe_rank0_print = lambda *args, **kwargs: print(*args, **kwargs)
    training_utils.rank0_print = safe_rank0_print

    return SimpleNamespace(
        Trainer=trainer_module.Trainer,
        TrainingArguments=training_utils.TrainingArguments,
        construct_revos_train=construct_module.construct_revos_train,
    )


def _try_load_dataset_runtime():
    dataset_module = importlib.import_module("easy_vlm.training.dataset")
    safe_rank0_print = lambda *args, **kwargs: print(*args, **kwargs)
    dataset_module.rank0_print = safe_rank0_print
    return SimpleNamespace(
        SFTDataset=dataset_module.SFTDataset,
        DataCollator=dataset_module.DataCollator,
    )


def _try_load_build_model():
    train_module = importlib.import_module("easy_vlm.train")
    return train_module.build_model


def _move_to_device(value: Any, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _classify_runtime_failure(exc: Exception) -> str:
    if isinstance(exc, (ImportError, ModuleNotFoundError, FileNotFoundError, PermissionError, OSError)):
        return "env_missing"

    message = str(exc)
    env_markers = (
        "No module named",
        "No such file or directory",
        "CUDA is not available",
        "Found no NVIDIA driver",
        "not compiled with CUDA",
        "libcuda",
        "NCCL",
        "cudnn",
        "triton",
        "weights/pretrain_weights",
    )
    if any(marker in message for marker in env_markers):
        return "env_missing"
    return "code_error"


def _inspect_named_video_query_params(named_params: Dict[str, nn.Parameter]) -> Dict[str, Any]:
    projector_names = [
        name
        for name, param in named_params.items()
        if "video_query_projector" in name and param.requires_grad
    ]
    alpha_names = [
        name
        for name, param in named_params.items()
        if "video_query_alpha" in name and param.requires_grad
    ]
    return {
        "projector_param_names": projector_names,
        "projector_found": len(projector_names) > 0,
        "alpha_param_names": alpha_names,
        "alpha_found": len(alpha_names) > 0,
    }


class _TinyOptimizerCheckModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(8, 8)
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))])
        self.mask_hidden_fcs = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))])
        self.video_query_projector = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
        self.video_query_alpha = nn.Parameter(torch.zeros(1))
        self.mask_queries = nn.Parameter(torch.zeros(4, 8))

    def forward(self, **kwargs):
        raise RuntimeError("Tiny optimizer-check model is not intended for forward passes.")


def _make_training_args(
    runtime,
    args: argparse.Namespace,
    output_dir: Path,
    ann_path: Optional[Path] = None,
):
    ann_list = [str(ann_path)] if ann_path is not None else ["dummy.json"]
    kwargs = dict(
        output_dir=str(output_dir),
        model_path=args.model_path,
        mask_decoder_model=args.mask_decoder_model,
        ann_path=ann_list,
        data_root=args.data_root,
        data_path_root="/",
        data_cache_dir=str(output_dir / "hf_cache"),
        max_seg_nums=args.max_seg_nums,
        seg_encoder="sam3",
        seg_decoder="sam3",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        remove_unused_columns=False,
        report_to=[],
        logging_steps=1,
        save_strategy="no",
        dataloader_num_workers=0,
        loss_reduction_scope="batch",
        average_tokens_across_devices=False,
        bf16=args.bf16,
        fp16=False,
        use_token_compression=False,
        attn_implementation=args.attn_implementation,
        llm_lr=0.0,
        projector_lr=0.0,
        vision_encoder_lr=0.0,
        sam_decoder_lr=args.sam_decoder_lr,
        sam_encoder_lr=None,
        lora_enable=False,
        max_frames=args.max_frames,
        fps=args.fps,
        model_max_length=args.model_max_length,
        mm_max_length=args.mm_max_length,
        skip_none=False,
        use_multi_objs=False,
    )
    arg_fields = getattr(runtime.TrainingArguments, "__dataclass_fields__", {})
    if "eval_strategy" in arg_fields:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    return runtime.TrainingArguments(**kwargs)


def _inspect_optimizer_groups(model: nn.Module, optimizer) -> Dict[str, Any]:
    named_params = dict(model.named_parameters())
    target_params = _inspect_named_video_query_params(named_params)
    projector_names = target_params["projector_param_names"]
    alpha_names = target_params["alpha_param_names"]
    param_id_to_name = {id(param): name for name, param in named_params.items()}

    group_summaries = []
    projector_group_names = []
    alpha_group_names = []
    projector_params_in_optimizer = set()
    alpha_params_in_optimizer = set()
    for group_idx, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", f"group_{group_idx}")
        group_param_names = []
        for param in group["params"]:
            param_name = param_id_to_name.get(id(param))
            if param_name is None:
                continue
            group_param_names.append(param_name)
            if "video_query_projector" in param_name:
                projector_params_in_optimizer.add(param_name)
                projector_group_names.append(group_name)
            if "video_query_alpha" in param_name:
                alpha_params_in_optimizer.add(param_name)
                alpha_group_names.append(group_name)

        group_summaries.append(
            {
                "index": group_idx,
                "name": group_name,
                "param_count": len(group["params"]),
                "video_query_projector_param_count": sum(
                    1 for name in group_param_names if "video_query_projector" in name
                ),
                "video_query_alpha_param_count": sum(
                    1 for name in group_param_names if "video_query_alpha" in name
                ),
            }
        )

    projector_all_in_optimizer = sorted(projector_params_in_optimizer) == sorted(projector_names)
    alpha_all_in_optimizer = sorted(alpha_params_in_optimizer) == sorted(alpha_names)
    return {
        **target_params,
        "projector_params_in_optimizer": sorted(projector_params_in_optimizer),
        "projector_all_in_optimizer": projector_all_in_optimizer,
        "projector_group_names": sorted(set(projector_group_names)),
        "alpha_params_in_optimizer": sorted(alpha_params_in_optimizer),
        "alpha_all_in_optimizer": alpha_all_in_optimizer,
        "alpha_group_names": sorted(set(alpha_group_names)),
        "group_summaries": group_summaries,
    }


def _print_optimizer_report(report: Dict[str, Any]) -> None:
    print(f"video_query_projector params found: {report['projector_found']}")
    print(f"video_query_projector param names: {report['projector_param_names']}")
    print(f"video_query_projector params in optimizer: {report['projector_params_in_optimizer']}")
    print(f"video_query_projector all in optimizer groups: {report['projector_all_in_optimizer']}")
    print(f"video_query_projector group names: {report['projector_group_names']}")
    print(f"video_query_alpha params found: {report['alpha_found']}")
    print(f"video_query_alpha param names: {report['alpha_param_names']}")
    print(f"video_query_alpha params in optimizer: {report['alpha_params_in_optimizer']}")
    print(f"video_query_alpha all in optimizer groups: {report['alpha_all_in_optimizer']}")
    print(f"video_query_alpha group names: {report['alpha_group_names']}")
    print(f"optimizer group count: {len(report['group_summaries'])}")
    for group in report["group_summaries"]:
        print(
            f"  group[{group['index']}]: name={group['name']} "
            f"param_count={group['param_count']} "
            f"video_query_projector_param_count={group['video_query_projector_param_count']} "
            f"video_query_alpha_param_count={group['video_query_alpha_param_count']}"
        )


def _run_optimizer_check(runtime, args: argparse.Namespace, tmp_path: Path):
    training_args = _make_training_args(runtime, args, tmp_path / "optimizer_check")
    fake_model = _TinyOptimizerCheckModel()
    trainer = runtime.Trainer(
        model=fake_model,
        args=training_args,
        train_dataset=[],
        data_collator=lambda x: x,
    )
    trainer.create_optimizer()
    optimizer = trainer.optimizer
    if optimizer is None:
        raise RuntimeError("trainer.create_optimizer() returned None")
    report = _inspect_optimizer_groups(fake_model, optimizer)
    return trainer, optimizer, report


def _run_real_train_step(
    runtime,
    args: argparse.Namespace,
    tmp_path: Path,
    train_json: Path,
) -> Tuple[str, Optional[float], Optional[str]]:
    try:
        print("[smoke] importing dataset runtime...")
        dataset_runtime = _try_load_dataset_runtime()
    except Exception as exc:
        failure_kind = _classify_runtime_failure(exc)
        return failure_kind, None, f"dataset runtime import failed: {exc}"

    try:
        print("[smoke] importing build_model...")
        build_model = _try_load_build_model()
    except Exception as exc:
        failure_kind = _classify_runtime_failure(exc)
        return failure_kind, None, f"build_model import failed: {exc}"

    try:
        print("[smoke] building real model...")
        training_args = _make_training_args(runtime, args, tmp_path / "real_step", ann_path=train_json)
        model, processor, seg_processor = build_model(training_args)
        print("[smoke] real model built.")
    except Exception as exc:
        failure_kind = _classify_runtime_failure(exc)
        return failure_kind, None, f"real model build failed: {exc}"

    try:
        named_params = dict(model.named_parameters())
        real_param_report = _inspect_named_video_query_params(named_params)
        print(f"[smoke] real model trainable video_query_projector params: {real_param_report['projector_param_names']}")
        print(f"[smoke] real model trainable video_query_alpha params: {real_param_report['alpha_param_names']}")
        if not real_param_report["projector_found"]:
            return "code_error", None, "real model build left video_query_projector frozen or missing"
        if not real_param_report["alpha_found"]:
            return "code_error", None, "real model build left video_query_alpha frozen or missing"

        print("[smoke] constructing dataset...")
        dataset = dataset_runtime.SFTDataset(
            model_config=model.config,
            processor=processor,
            seg_processor=seg_processor,
            model_max_length=args.model_max_length,
            mm_max_length=args.mm_max_length,
            fps=args.fps,
            max_frames=args.max_frames,
            dataloader_num_workers=0,
            data_args=training_args,
            requires_length=False,
            use_multi_objs=False,
        )
        if len(dataset) == 0:
            return "code_error", None, "constructed dataset is empty"

        print("[smoke] materializing dataset instances...")
        instances = [dataset[idx] for idx in range(min(len(dataset), args.max_samples))]
        print("[smoke] collating batch...")
        batch = dataset_runtime.DataCollator(processor=processor, sequence_packing=False)(instances)

        print("[smoke] creating trainer/optimizer for real model...")
        trainer = runtime.Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=dataset_runtime.DataCollator(processor=processor, sequence_packing=False),
        )
        trainer.create_optimizer()
        optimizer = trainer.optimizer
        if optimizer is None:
            return "code_error", None, "trainer.create_optimizer() returned None for real model"

        device = torch.device(args.device)
        print(f"[smoke] moving model to {device}...")
        model.to(device)
        model.train()
        print("[smoke] moving batch to device...")
        batch = _move_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        print("[smoke] running forward...")
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            return "code_error", None, f"loss is invalid: {loss}"
        print(f"[smoke] forward ok, loss={loss.item():.6f}")
        print("[smoke] running backward...")
        loss.backward()
        print("[smoke] optimizer.step()...")
        optimizer.step()
        return "passed", float(loss.item()), None
    except Exception as exc:
        failure_kind = _classify_runtime_failure(exc)
        return failure_kind, None, f"real train step failed: {exc}\n{traceback.format_exc()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REVoS video-train smoke test.")
    parser.add_argument("--model-path", type=str, default="weights/pretrain_weights")
    parser.add_argument("--mask-decoder-model", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="/root/InstructSAM/video_dataset/revos")
    parser.add_argument(
        "--meta-path",
        type=str,
        default="/root/InstructSAM/video_dataset/revos/meta_expressions_train_.json",
    )
    parser.add_argument(
        "--mask-dict-path",
        type=str,
        default="/root/InstructSAM/video_dataset/revos/mask_dict.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/root/InstructSAM/outputs/revos_train/revos_train_smoke.json",
    )
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--mm-max-length", type=int, default=2048)
    parser.add_argument("--max-seg-nums", type=int, default=10)
    parser.add_argument("--sam-decoder-lr", type=float, default=1e-5)
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    runtime = _load_runtime(repo_root)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    stats, video_meta_output_path = runtime.construct_revos_train(
        meta_path=Path(args.meta_path),
        mask_dict_path=Path(args.mask_dict_path),
        data_root=Path(args.data_root),
        output_path=output_json,
        max_samples=args.max_samples,
        skip_missing_videos=True,
    )
    print(f"Constructed subset at {output_json}")
    print(f"Constructed video metadata sidecar at {video_meta_output_path}")
    print(f"Subset stats: {stats}")

    with tempfile.TemporaryDirectory(prefix="revos_smoke_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        trainer, optimizer, optimizer_report = _run_optimizer_check(runtime, args, tmp_path)
        if optimizer is None:
            raise RuntimeError("Optimizer check failed to create an optimizer")
        _print_optimizer_report(optimizer_report)
        if not optimizer_report["projector_found"]:
            raise RuntimeError("video_query_projector parameters were not found in the smoke-check model")
        if not optimizer_report["alpha_found"]:
            raise RuntimeError("video_query_alpha parameter was not found in the smoke-check model")
        if not optimizer_report["projector_all_in_optimizer"]:
            raise RuntimeError("Not all video_query_projector parameters were placed into optimizer groups")
        if not optimizer_report["alpha_all_in_optimizer"]:
            raise RuntimeError("video_query_alpha was not placed into optimizer groups")

        expected_group_names = {"sam_decoder", "sam_decoder_nodecay"}
        if not set(optimizer_report["projector_group_names"]).issubset(expected_group_names):
            raise RuntimeError(
                "video_query_projector parameters landed in unexpected optimizer groups: "
                f"{optimizer_report['projector_group_names']}"
            )
        if not set(optimizer_report["alpha_group_names"]).issubset(expected_group_names):
            raise RuntimeError(
                "video_query_alpha landed in unexpected optimizer groups: "
                f"{optimizer_report['alpha_group_names']}"
            )
        if not set(optimizer_report["alpha_group_names"]).intersection(optimizer_report["projector_group_names"]):
            raise RuntimeError(
                "video_query_alpha did not share a training optimizer path with video_query_projector: "
                f"alpha_groups={optimizer_report['alpha_group_names']} "
                f"projector_groups={optimizer_report['projector_group_names']}"
            )

        real_step_status, loss_value, real_step_error = _run_real_train_step(
            runtime=runtime,
            args=args,
            tmp_path=tmp_path,
            train_json=output_json,
        )
        print(f"real forward/backward/step status: {real_step_status}")
        if real_step_status == "passed":
            print(f"real train-step loss: {loss_value:.6f}")
        elif real_step_status == "env_missing":
            print(f"real train-step skipped_environment: {real_step_error}")
        else:
            print(f"real train-step code_error: {real_step_error}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
