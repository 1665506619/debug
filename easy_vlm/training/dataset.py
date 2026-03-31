import os
import traceback
import random
from math import ceil
from typing import Optional, List, Dict, Any
import copy
import hashlib
import json
import pickle
import torch
import numpy as np
from PIL import Image, ImageOps
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import ProcessorMixin, logging, PretrainedConfig
from .utils import rank0_print, SEG_IMAGE_QUESTIONS_PHRASE, SEG_VIDEO_QUESTIONS_PHRASE, SEG_IMAGE_QUESTIONS_OCR, SEG_IMAGE_QUESTIONS_PHRASE_MULTI
from .mm_utils import annToMask, resize_nearest_like_torch, iou_mask
from ..constants import SEG_TOKEN, REF_START_TOKEN, REF_END_TOKEN, SEG_START_TOKEN, SEG_END_TOKEN, IGNORE_INDEX, MAX_PHRASE_NUM, MAX_OBJ_NUM
from ..models.utils import load_video

logger = logging.get_logger(__name__)

VIDEO_FILE_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".gif", ".webm")

data_replace_dict = {
    "/mnt/damovl/MEDIA/IMAGE/Objects365/raw/Objects365/data/train/":"object365/images/train/",
    "SA-1B/": "/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/SA-1B/"
}

_REVOS_VIDEO_META_CACHE: Dict[str, Dict[str, Any]] = {}
_REVOS_MASK_DICT_CACHE: Dict[str, Dict[str, Any]] = {}
_REVOS_OBJECT_MASK_CACHE: Dict[tuple[str, int, str], Dict[str, Any]] = {}

def _get_rope_index_qwen3_vl(
    model_config: PretrainedConfig,
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

    # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    spatial_merge_size = model_config.vision_config.spatial_merge_size
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids


def _get_rope_index(
    model_config: PretrainedConfig,
    input_ids: torch.LongTensor,
    **kwargs,
) -> torch.Tensor:
    if model_config.model_type == "qwen3_vl":
        position_ids = _get_rope_index_qwen3_vl(
            model_config=model_config,
            input_ids=input_ids,
            **kwargs,
        )
    elif model_config.model_type == "video_llama_3":
        position_ids = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported model: {model_config.model_type}")
    return position_ids

def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


def _video_mask_sort_key(key):
    key_str = str(key)
    if key_str.isdigit():
        return (0, int(key_str))
    return (1, key_str)

class DataCollator(object):
    def __init__(
        self,
        processor: ProcessorMixin,
        sequence_packing: bool,
    ):
        self.processor = processor
        self.sequence_packing = sequence_packing

    def _collate_mm_inputs(self, instances):
        mm_input_names = set(
            self.processor.image_processor.model_input_names + self.processor.video_processor.model_input_names
        )

        mm_inputs = {}
        for key in mm_input_names:
            data_list = [
                instance[key]
                for instance in instances
                if key in instance and instance[key] is not None
            ]
            if len(data_list) > 0:
                mm_inputs[key] = torch.cat(data_list, dim=0)

        return mm_inputs

    def _collate_fn_packing(self, instances):
        input_ids, position_ids, labels = [], [], []
        for instance in instances:
            input_ids.append(instance["input_ids"])
            if "position_ids" in instance:
                position_ids.append(instance["position_ids"])
            else:
                position_ids.append(torch.arange(instance["input_ids"].size(-1)).unsqueeze(0))
            tmp_labels = instance["labels"].clone()
            tmp_labels[..., 0] = -100
            labels.append(tmp_labels)

        batch = {
            "data_indices": [instance["data_index"] for instance in instances],
            "input_ids": torch.cat(input_ids, dim=-1),
            "position_ids": torch.cat(position_ids, dim=-1),
            "labels": torch.cat(labels, dim=-1),
            **self._collate_mm_inputs(instances),
        }

        batch["masks"] = [x["masks"] for x in instances]
        batch["sam_images"] = [x["sam_images"] for x in instances]
        batch["sam_size"] = [x["sam_size"] for x in instances]
        batch["masks_valid"] = [x["mask_valid"] for x in instances]
        batch["mask_type"] = [x["mask_type"] for x in instances]
        batch["mask_ids"] = [x["mask_ids"] for x in instances]
        batch["modalities"] = [x["sample_modality"] for x in instances]

        return batch

    def _collate_fn_padding(self, instances):
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)

        batch = dict(
            data_indices=[instance["data_index"] for instance in instances],
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.processor.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance and instance["pixel_values"] is not None
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
            and instance["pixel_values_videos"] is not None
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance and instance["image_grid_thw"] is not None
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance and instance["video_grid_thw"] is not None
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        
        batch["phrase_ids"] = [x["phrase_ids"] for x in instances]
        batch["masks"] = [x["masks"] for x in instances]
        batch["sam_images"] = [x["sam_images"] for x in instances]
        batch["sam_size"] = [x["sam_size"] for x in instances]
        batch["masks_valid"] = [x["mask_valid"] for x in instances]
        batch["mask_type"] = [x["mask_type"] for x in instances]
        batch["mask_ids"] = [x["mask_ids"] for x in instances]
        batch["modalities"] = [x["sample_modality"] for x in instances]

        return batch

    def __call__(self, instances: List[Dict[str, Any]]):
        if self.sequence_packing:
            return self._collate_fn_packing(instances)
        return self._collate_fn_padding(instances)


def preprocess_data_list(data_list, data_name="Dataset", max_seg_num=10):
    """
    Preprocess data list by splitting annotations according to mask/bbox/point limits.
    Print dataset size before and after processing, and per-sample split statistics.
    """

    processed_data = []

    rank0_print(f"[Preprocess] {data_name} Raw data size: {len(data_list)}")
    for idx, data in enumerate(data_list):
        original_len = len(processed_data)

        if "annotations" in data:
            data = copy.deepcopy(data)
            data["annotation"] = data.pop("annotations")

        if "annotation" not in data:
            processed_data.append(data)
            continue

        annotations = data["annotation"]
        random.shuffle(annotations)

        if "image" in data:
            MAX_MASK_NUM = 10
        elif "video" in data and len(data.get("frame_idx", [])) >= 32:
            MAX_MASK_NUM = 1
        elif "video" in data and len(data.get("frame_idx", [])) >= 16:
            MAX_MASK_NUM = 2
        else:
            MAX_MASK_NUM = 5

        mask_count = 0
        current_annotations = []

        for ann in annotations:
            if ann.get("mask") is None and ann.get("bbox") is None and ann.get("point") is None:
                current_annotations.append(ann)
                continue

            if "mask" in ann and len(ann["mask"]) > max_seg_num:
                continue
            if "bbox" in ann and len(ann["bbox"]) > max_seg_num:
                continue

            if "mask" in ann:
                mask_count += 1
            elif "bbox" in ann:
                mask_count += 1
            elif "point" in ann:
                mask_count += 1

            current_annotations.append(ann)

            if mask_count >= MAX_MASK_NUM:
                split_data = copy.deepcopy(data)
                split_data["annotation"] = current_annotations.copy()
                processed_data.append(split_data)

                current_annotations.clear()
                mask_count = 0

        if current_annotations:
            split_data = copy.deepcopy(data)
            split_data["annotation"] = current_annotations
            processed_data.append(split_data)

    rank0_print(f"[Preprocess] {data_name} Processed data size: {len(processed_data)}")

    return processed_data



class SFTDataset(Dataset):
    def __init__(
        self,
        model_config: PretrainedConfig,
        processor: ProcessorMixin,
        seg_processor,
        model_max_length: int,
        mm_max_length: int,
        fps: int,
        max_frames: int,
        dataloader_num_workers: Optional[int],
        data_args: str,
        requires_length: bool = False,
        mask_size: int = 288,
        use_multi_objs: bool = True,
    ):
        self.model_config = model_config
        self.processor = processor
        self.seg_processor = seg_processor
        self.model_max_length = model_max_length
        self.mm_max_length = mm_max_length
        self.fps = fps
        self.max_frames = max_frames
        self.data_root = data_args.data_root
        self.max_seg_nums = data_args.max_seg_nums
        self.mask_size = mask_size
        self.use_multi_objs = use_multi_objs
        self.skip_none = data_args.skip_none
        output_dir = data_args.output_dir

        self._dataset = self._load_data(data_args.ann_path, data_args.data_cache_dir, data_args)

    @property
    def modality_lengths(self):
        length_list = []
        for data_dict in self._dataset:
            mask_num = 0
            if "annotation" in data_dict and data_dict["annotation"] is not None:
                if isinstance(data_dict["annotation"], str):
                    data_dict["annotation"] = json.loads(data_dict["annotation"])
                for ann in data_dict["annotation"]:
                    if "ann" in ann and ann["ann"] is not None:
                        mask_num += 1
            elif data_dict.get("source") == "revos_train" and data_dict.get("anno_ids") is not None:
                mask_num += len(data_dict["anno_ids"])
            elif "mask" in data_dict and data_dict["mask"] is not None:
                mask_num += len(data_dict['mask'])
            if mask_num == 0:
                mask_num = -1
            length_list.append(mask_num)

        return length_list

    def _load_data(self, data_path, cache_dir, data_args):
        def load_data_list(path):
            data_paths = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) == 1:
                        p, n = parts[0], 1
                    else:
                        p, n = parts[0], int(parts[1])
                    data_paths.extend([p] * n)
            return data_paths
        
        need_preprocess = True
        if '.txt' in data_path[0]:
            need_preprocess = True
            data_path = load_data_list(data_path[0])

        list_data_dict = []
        for d in data_path:
            rank0_print(f'begin load {d}...')
            source_json_path = os.path.join(data_args.data_path_root, d)
            dataset = load_dataset('json', data_files=source_json_path, cache_dir=cache_dir)['train']
            if "__source_json_path" not in dataset.column_names:
                dataset = dataset.add_column("__source_json_path", [source_json_path] * len(dataset))
            list_data_dict.append(dataset)
            # data_dict_inner = []
            # if d.endswith(".json"):
            #     data_dict_inner = json.load(open(os.path.join(data_args.data_path_root,d), "r"))
            # elif d.endswith(".jsonl"):
            #     with open(os.path.join(data_args.data_path_root,d), "r", encoding="utf-8") as fp:
            #         for line in fp:
            #             line = line.strip()
            #             obj = json.loads(line)
            #             data_dict_inner.append(obj)
            # else:
            #     raise Exception(f"Unsupported file format (<{d}>)!!!")
            
            # if need_preprocess:
            #     data_dict_inner = preprocess_data_list(data_dict_inner, d, self.max_seg_nums)
            # list_data_dict.extend(data_dict_inner)
        return concatenate_datasets(list_data_dict)

    def _is_compact_revos_train_sample(self, data_dict):
        return (
            data_dict.get("source") == "revos_train"
            and data_dict.get("source_dataset") == "revos"
            and data_dict.get("video") is None
            and data_dict.get("annotation") is None
            and data_dict.get("video_id") is not None
            and data_dict.get("anno_ids") is not None
            and data_dict.get("expression") is not None
        )

    def _resolve_revos_aux_path(self, data_dict, field_name, default_filename=None):
        path_value = data_dict.get(field_name)
        if path_value is None:
            if default_filename is None:
                return None
            path_value = default_filename

        if os.path.isabs(path_value):
            return path_value

        cwd_resolved = os.path.abspath(path_value)
        if os.path.exists(cwd_resolved):
            return cwd_resolved

        direct_resolved = self._resolve_local_path(path_value)
        if os.path.exists(direct_resolved):
            return direct_resolved

        source_json_path = data_dict.get("__source_json_path")
        if source_json_path is not None:
            source_resolved = os.path.join(os.path.dirname(source_json_path), path_value)
            if os.path.exists(source_resolved):
                return source_resolved
            return source_resolved
        return direct_resolved

    def _load_revos_video_meta(self, path):
        resolved_path = os.path.abspath(path)
        if resolved_path not in _REVOS_VIDEO_META_CACHE:
            with open(resolved_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "videos" in payload:
                payload = payload["videos"]
            _REVOS_VIDEO_META_CACHE[resolved_path] = payload
        return _REVOS_VIDEO_META_CACHE[resolved_path]

    def _load_revos_mask_dict(self, path):
        resolved_path = os.path.abspath(path)
        if resolved_path not in _REVOS_MASK_DICT_CACHE:
            with open(resolved_path, "r", encoding="utf-8") as f:
                _REVOS_MASK_DICT_CACHE[resolved_path] = json.load(f)
        return _REVOS_MASK_DICT_CACHE[resolved_path]

    def _build_revos_object_mask_dict(self, mask_dict_path, anno_id, frame_names, mask_dict):
        frame_key = "\t".join(frame_names)
        cache_key = (os.path.abspath(mask_dict_path), int(anno_id), frame_key)
        if cache_key not in _REVOS_OBJECT_MASK_CACHE:
            object_masks = mask_dict[str(anno_id)]
            if len(object_masks) != len(frame_names):
                raise ValueError(
                    f"REVoS anno_id={anno_id} has {len(object_masks)} masks but video has {len(frame_names)} frames"
                )
            _REVOS_OBJECT_MASK_CACHE[cache_key] = {
                frame_name: mask_ann
                for frame_name, mask_ann in zip(frame_names, object_masks)
            }
        return _REVOS_OBJECT_MASK_CACHE[cache_key]

    def _resolve_compact_revos_train_sample(self, data_dict):
        video_meta_path = self._resolve_revos_aux_path(
            data_dict,
            "revos_video_meta_file",
            default_filename="revos_train_sft_videos.json",
        )
        if video_meta_path is None:
            raise ValueError("Compact REVoS sample is missing video metadata sidecar path")
        video_meta_dict = self._load_revos_video_meta(video_meta_path)

        video_id = data_dict["video_id"]
        if video_id not in video_meta_dict:
            raise KeyError(f"video_id={video_id} is missing from {video_meta_path}")
        video_meta = video_meta_dict[video_id]
        frame_names = list(video_meta["frame_names"])

        mask_dict_path = self._resolve_revos_aux_path(
            data_dict,
            "revos_mask_dict_path",
        )
        if mask_dict_path is None:
            mask_dict_path = os.path.join(self.data_root, "mask_dict.json")
        mask_dict = self._load_revos_mask_dict(mask_dict_path)

        annotation_ann = [
            self._build_revos_object_mask_dict(
                mask_dict_path=mask_dict_path,
                anno_id=anno_id,
                frame_names=frame_names,
                mask_dict=mask_dict,
            )
            for anno_id in data_dict["anno_ids"]
        ]

        resolved = dict(data_dict)
        resolved["video"] = [f"{video_meta['video_rel_path']}/{frame_name}.jpg" for frame_name in frame_names]
        resolved["frame_names"] = frame_names
        resolved["annotation"] = [
            {
                "type": "phrase",
                "text": data_dict["expression"],
                "ann_type": "mask",
                "ann": annotation_ann,
            }
        ]
        resolved["height"] = data_dict.get("height", video_meta.get("height"))
        resolved["width"] = data_dict.get("width", video_meta.get("width"))
        return resolved

    def _resolve_local_path(self, path, apply_replace=False):
        if path is None:
            return None
        resolved_path = path
        if apply_replace:
            for key in data_replace_dict:
                if key in resolved_path:
                    resolved_path = resolved_path.replace(key, data_replace_dict[key])
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(self.data_root, resolved_path)
        return resolved_path

    def _is_video_file(self, path):
        return isinstance(path, str) and path.lower().endswith(VIDEO_FILE_EXTS)

    def _to_pil_image(self, image):
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(f"Expected a 3D video frame array, got shape {image.shape}")
            if image.shape[0] in {1, 3}:
                image = np.transpose(image, (1, 2, 0))
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        raise ValueError(f"Unsupported frame type for SAM preprocessing: {type(image)}")

    def _build_sam_image(self, image_source=None):
        if image_source is None:
            sam_size = (1008, 1008)
            return torch.zeros(3, sam_size[0], sam_size[1]), sam_size

        if isinstance(image_source, str):
            image = Image.open(image_source)
        else:
            image = image_source
        image = self._to_pil_image(image)
        sam_inputs = self.seg_processor(image)
        return sam_inputs["pixel_values"][0], sam_inputs.original_sizes[0]

    def _resolve_image_path(self, image_file):
        if image_file.startswith("train2017"):
            image_file = "coco/" + image_file
        return self._resolve_local_path(image_file, apply_replace=True)

    def _select_frame_paths(self, frame_paths, frame_idx):
        if frame_idx is None:
            return frame_paths
        try:
            return [frame_paths[i] for i in frame_idx]
        except IndexError as exc:
            raise ValueError(
                f"frame_idx contains out-of-range indices for a video with {len(frame_paths)} frames: {frame_idx}"
            ) from exc

    def _resolve_video_input(self, data_dict):
        video_value = data_dict["video"]
        frame_idx = data_dict.get("frame_idx")
        strict_video_mask_match = False

        if isinstance(video_value, list):
            frame_paths = [self._resolve_local_path(frame) for frame in video_value]
            frame_paths = self._select_frame_paths(frame_paths, frame_idx)
            if len(frame_paths) == 0:
                raise ValueError("Video frame list is empty after applying frame_idx")
            video_content = frame_paths
            sampled_frames, metadata = load_video(
                frame_paths,
                fps=self.fps,
                max_frames=self.max_frames,
            )
            sampled_indices = list(metadata.frames_indices)
            sampled_paths = [frame_paths[i] for i in sampled_indices]
            frame_candidates = []
            if frame_idx is not None:
                original_frame_idx = [frame_idx[i] for i in sampled_indices]
                frame_candidates.extend(
                    [original_frame_idx[0], str(original_frame_idx[0])]
                )
            frame_stem = os.path.splitext(os.path.basename(sampled_paths[0]))[0]
            frame_candidates.append(frame_stem)
        elif isinstance(video_value, str):
            video_path = self._resolve_local_path(video_value)
            if self._is_video_file(video_path):
                video_content = video_path
                strict_video_mask_match = True
                sampled_frames, metadata = load_video(
                    video_path,
                    fps=self.fps,
                    max_frames=self.max_frames,
                )
                sampled_indices = list(metadata.frames_indices)
                sampled_paths = [None] * len(sampled_indices)
                frame_candidates = [sampled_indices[0], str(sampled_indices[0])]
            elif os.path.isdir(video_path):
                frame_files = sorted(
                    [
                        os.path.join(video_path, frame_file)
                        for frame_file in os.listdir(video_path)
                        if frame_file.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
                if len(frame_files) == 0:
                    raise ValueError(f"No frames found under video directory: {video_path}")
                frame_files = self._select_frame_paths(frame_files, frame_idx)
                video_content = frame_files
                sampled_frames, metadata = load_video(
                    frame_files,
                    fps=self.fps,
                    max_frames=self.max_frames,
                )
                sampled_indices = list(metadata.frames_indices)
                sampled_paths = [frame_files[i] for i in sampled_indices]
                frame_candidates = []
                if frame_idx is not None:
                    original_frame_idx = [frame_idx[i] for i in sampled_indices]
                    frame_candidates.extend(
                        [original_frame_idx[0], str(original_frame_idx[0])]
                    )
                frame_stem = os.path.splitext(os.path.basename(sampled_paths[0]))[0]
                frame_candidates.append(frame_stem)
            else:
                raise ValueError(
                    f"Unsupported video input: expected frame list, video file, or frame directory, got {video_value}"
                )
        else:
            raise ValueError(
                f"Unsupported video input type: {type(video_value)}"
            )

        if len(sampled_frames) == 0:
            raise ValueError(f"No video frames could be sampled from {video_value}")

        sam_frame = self._to_pil_image(sampled_frames[0])
        return video_content, sam_frame, frame_candidates, strict_video_mask_match

    def _infer_video_mask_size(self, video_mask_dict, data_dict, fallback_image=None):
        if data_dict.get("height") is not None and data_dict.get("width") is not None:
            return data_dict["height"], data_dict["width"]

        for mask_ann in video_mask_dict.values():
            if mask_ann is not None:
                size = mask_ann.get("size", None)
                if size is not None:
                    return size[0], size[1]

        if fallback_image is not None:
            return fallback_image.size[1], fallback_image.size[0]
        raise ValueError("Unable to infer video mask size from annotation or frames")

    def _select_video_mask_frame(
        self,
        video_mask_dict,
        data_dict,
        frame_candidates,
        fallback_image=None,
        strict_match=False,
    ):
        if not isinstance(video_mask_dict, dict):
            raise ValueError(
                f"Expected video mask annotation to be a dict indexed by frame, got {type(video_mask_dict)}"
            )

        selected_key = None
        for candidate in frame_candidates:
            if candidate in video_mask_dict:
                selected_key = candidate
                break
            if isinstance(candidate, int) and str(candidate) in video_mask_dict:
                selected_key = str(candidate)
                break
            if isinstance(candidate, int):
                for key in video_mask_dict.keys():
                    key_str = str(key)
                    if key_str.isdigit() and int(key_str) == candidate:
                        selected_key = key
                        break
                if selected_key is not None:
                    break
            if isinstance(candidate, str) and candidate.isdigit():
                int_candidate = int(candidate)
                if int_candidate in video_mask_dict:
                    selected_key = int_candidate
                    break

        if selected_key is None:
            if strict_match:
                raise ValueError(
                    "Unable to align video supervision to the sampled frame. "
                    f"Tried frame candidates {frame_candidates}, but annotation keys are "
                    f"{list(video_mask_dict.keys())[:8]}."
                )
            sorted_keys = sorted(video_mask_dict.keys(), key=_video_mask_sort_key)
            if len(sorted_keys) == 0:
                raise ValueError("Video mask annotation dict is empty")
            selected_key = sorted_keys[0]

        mask_ann = video_mask_dict[selected_key]
        mask_h, mask_w = self._infer_video_mask_size(
            video_mask_dict,
            data_dict,
            fallback_image=fallback_image,
        )
        if mask_ann is None:
            return np.zeros((mask_h, mask_w), dtype=np.uint8)
        return annToMask(mask_ann, mask_h, mask_w)

    def _convert_conversation(self, data_dict):
        data_folder = self.data_root

        mask_ids = []
        masks = []
        mask_type = 0 # 0: mask, 1: bbox, 2: point
        mask_valid = False
        new_conversation = []
        new_contents = []
        sam_frame_source = None
        video_frame_candidates = []
        strict_video_mask_match = False
        strict_video_mask_match = False
        phrase_str = ''

        if 'height' in data_dict:
            h = data_dict['height']
            w = data_dict['width']
        else:
            h = None
            w = None

        if 'image' in data_dict and data_dict['image'] is not None:
            modal = 'image'
            image_file = self._resolve_image_path(data_dict['image'])
            new_contents.append({"type": "image", "image": image_file})
            sam_frame_source = image_file
            # new_contents.append({"type": "image", "image": os.path.join(data_folder, image_file)})
            # images.append(os.path.join(data_folder, image_file))
                
            if "mask" in data_dict and data_dict["mask"] is not None and len(data_dict["mask"])>0: # mask
                for msk in data_dict["mask"]:
                    mask = annToMask(msk, h, w)
                    mask_ids.append([len(masks)])
                    masks.append(np.expand_dims(mask, axis=0))
                mask_valid = True

        elif 'video' in data_dict and data_dict['video'] is not None:
            modal = 'video'
            video_content, sam_frame_source, video_frame_candidates, strict_video_mask_match = self._resolve_video_input(data_dict)
            new_contents.append({"type": "video", "video": video_content})

        else:
            modal = 'text'
        
        if "annotation" in data_dict and data_dict["annotation"] is not None: # grounding data
            if isinstance(data_dict["annotation"], str):
                data_dict["annotation"] = json.loads(data_dict["annotation"])
            conversation = []
            annotations = data_dict["annotation"]
            random.shuffle(annotations)
            phrase_num = 0
            obj_num = 0
            if self.use_multi_objs:
                if len(annotations)<=2:
                    multi_idx = -1
                elif modal=='image' and len(annotations)>0 and annotations[0]["type"]=='phrase':
                    multi_idx = len(annotations)-1
                    masks_multi_obj = []
                    while multi_idx>=0:
                        ann = annotations[multi_idx]
                        if ann["ann_type"]=="mask" and ann["ann"] is not None:
                            mask_all_multi_obj = np.zeros((1, h, w)) 
                            for msk in ann["ann"]:
                                msk = json.loads(msk)
                                mask_cur = np.expand_dims(annToMask(msk, h, w), axis=0) #[1,h,w]
                                masks_multi_obj.append(mask_cur)
                                mask_all_multi_obj = np.maximum(mask_all_multi_obj, mask_cur)
                                phrase_cur_multi_obj = annotations[multi_idx]["text"]
                            break
                        multi_idx -= 1
                else:
                    multi_idx = -1
            else:
                multi_idx = -1
            for i,annotation in enumerate(annotations):
                if i==multi_idx:
                    continue
                if self.skip_none and annotation["ann"] is None: # 不训none
                    continue
                ann_len = len(annotation["ann"]) if "ann" in annotation and annotation["ann"] is not None else 0
                if ann_len>self.max_seg_nums:
                    continue
                if obj_num+ann_len>MAX_OBJ_NUM or phrase_num+1>MAX_PHRASE_NUM:
                    break
                if annotation['type']=='phrase':
                    phrase = annotation['text'].strip()
                    phrase = phrase.lower()[0]+phrase[1:]
                    if phrase.endswith('.'):
                        phrase = phrase[:-1]  
                    if modal=='image':
                        new_contents.append({'type': 'text', 'text': random.choice(SEG_IMAGE_QUESTIONS_PHRASE).format(phrase=phrase)})
                    else:   
                        new_contents.append({'type': 'text', 'text': random.choice(SEG_VIDEO_QUESTIONS_PHRASE).format(phrase=phrase)})
                elif annotation['type']=='OCR':
                    phrase = annotation['text']
                    if modal=='image':
                        new_contents.append({'type': 'text', 'text': random.choice(SEG_IMAGE_QUESTIONS_OCR).format(phrase=phrase)})
                    else:
                        raise ValueError("OCR annotations are not supported for video samples yet")
                elif annotation['type']=='sentence':
                    phrase = annotation['category']
                    new_contents.append({'type': 'text', 'text': annotation['text']})
                else:
                    raise ValueError(f"No phrase or sentence in the annotation")

                message = {"role": "user", "content": new_contents}
                new_conversation.append(message)
                new_contents = []

                mask_ids_inner = []
                if annotation["ann_type"]=="mask": #mask
                    if annotation["ann"] is None:
                        if modal=='image':
                            new_contents.append({'type': 'text', 'text': f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}"})
                            phrase_str += f"{REF_START_TOKEN}{phrase}"
                            phrase_num +=1
                            message = {"role": "assistant", "content": new_contents}
                            new_conversation.append(message)
                            new_contents = []
                            mask_cur = np.zeros((1, h, w)) #[1,h,w]
                            mask_ids.append([len(masks)])
                            masks.append(mask_cur)
                        else:
                            raise ValueError("Null masks are not supported for video samples yet")
                        continue
                    else:
                        mask_valid = True
                        new_contents.append({'type': 'text', 'text': f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}"})
                        phrase_str += f"{REF_START_TOKEN}{phrase}"
                        phrase_num +=1
                        obj_num += ann_len

                    # if 'segmentation_file' in data_dict:
                    #     mask_json = json.load(open(os.path.join(data_folder, data_dict['segmentation_file'])))
                    mask_all = np.zeros((1, h, w)) if modal == 'image' else None
                    for msk in annotation["ann"]:
                        if isinstance(msk, str):
                            msk = json.loads(msk)
                        if modal=='image':
                            mask_cur = np.expand_dims(annToMask(msk, h, w), axis=0) #[1,h,w]
                            mask_all = np.maximum(mask_all, mask_cur)
                        else:
                            mask_cur = np.expand_dims(
                                self._select_video_mask_frame(
                                    msk,
                                    data_dict,
                                    video_frame_candidates,
                                    fallback_image=sam_frame_source,
                                    strict_match=strict_video_mask_match,
                                ),
                                axis=0,
                            )

                        mask_ids_inner.append(len(masks))
                        masks.append(mask_cur)
                    mask_ids.append(mask_ids_inner)
                    # add multi phrase format
                    if multi_idx!=-1 and mask_all is not None and iou_mask(mask_all, mask_all_multi_obj)<0.05:
                        mask_ids_inner = []
                        for idx_ in range(len(masks_multi_obj)):
                            mask_ids_inner.append(len(masks)+idx_)
                        masks += masks_multi_obj
                        mask_ids.append(mask_ids_inner)
                        new_contents[-1]['text'] += f", {REF_START_TOKEN}{phrase_cur_multi_obj}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}"
                        phrase_str += f"{REF_START_TOKEN}{phrase_cur_multi_obj}"
                        phrase_num += 1
                        obj_num += ann_len
                        new_conversation[-1]['content'][-1]['text'] = random.choice(SEG_IMAGE_QUESTIONS_PHRASE_MULTI).format(phrase1=phrase, phrase2=phrase_cur_multi_obj)
                        multi_idx = -1

                elif annotation["ann_type"]=="point":
                    mask_type = 2
                    pass
                elif annotation["ann_type"]=="bbox":
                    mask_type = 1
                    new_contents.append({'type': 'text', 'text': f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}"})
                    phrase_str += f"{REF_START_TOKEN}{phrase}"
                    phrase_num +=1
                    obj_num += ann_len
                    for bbox in annotation["ann"]:
                        if isinstance(bbox, str):
                            bbox = json.loads(bbox)
                        if modal=='image':
                            mask = np.zeros((h, w))
                            x1, y1, wb, hb = bbox
                            x1 = int(max(x1,0))
                            y1 = int(max(y1,0))
                            wb = int(min(wb, w - x1))
                            hb = int(min(hb, h - y1))
                            mask[y1:y1+hb, x1:x1+wb] = 1
                            mask = np.expand_dims(mask, axis=0) #[1,h,w]
                        else:
                            raise ValueError("Bounding box annotations are not supported for video samples yet")
                        mask_ids_inner.append(len(masks))
                        masks.append(mask)
                    mask_ids.append(mask_ids_inner)

                message = {"role": "assistant", "content": new_contents}
                new_conversation.append(message)
                new_contents = []
                # if obj_num>=MAX_OBJ_NUM or phrase_num>=MAX_PHRASE_NUM:
                #     break

        if len(masks)>0:
            masks = np.array(masks)
            masks = resize_nearest_like_torch(masks, self.mask_size, self.mask_size)
            masks = torch.from_numpy(masks)
        else:
            masks = None

        if 'conversations' in data_dict and data_dict['conversations'] is not None:
            if isinstance(data_dict['conversations'], str):
                data_dict['conversations'] = json.loads(data_dict['conversations'])
            for idx, conv in enumerate(data_dict['conversations']):
                new_contents.append({'type': 'text', 'text': conv['value'].replace('<image>','').replace('<video>','').strip()})
                if idx%2==0:
                    message = {"role": "user", "content": new_contents}
                else:
                    message = {"role": "assistant", "content": new_contents}
                new_conversation.append(message)
                new_contents = []
                # phrase_str+=conv['value']

        sam_images, sam_size = self._build_sam_image(sam_frame_source)

        # if not mask_valid:
        # print(new_conversation)
        # print('**************')
        # print(phrase_str)
        # print('==============')
        if len(new_conversation)==0:
            print(data_dict)
            print('==============')

        return new_conversation, sam_images, sam_size, masks, mask_ids, mask_valid, mask_type, phrase_str

    def _convert_conversation_instseg(self, data_dict):
        data_folder = self.data_root

        mask_ids = []
        masks = []
        mask_type = 0 # 0: mask, 1: bbox, 2: point
        mask_valid = False
        new_conversation = []
        new_contents = []
        sam_frame_source = None
        video_frame_candidates = []
        phrase_str = ''

        if 'height' in data_dict:
            h = data_dict['height']
            w = data_dict['width']
        else:
            h = None
            w = None

        if 'image' in data_dict and data_dict['image'] is not None:
            modal = 'image'
            image_file = self._resolve_image_path(data_dict['image'])
            new_contents.append({"type": "image", "image": image_file})
            sam_frame_source = image_file

        elif 'video' in data_dict and data_dict['video'] is not None:
            modal = 'video'
            video_content, sam_frame_source, video_frame_candidates, strict_video_mask_match = self._resolve_video_input(data_dict)
            new_contents.append({"type": "video", "video": video_content})

        else:
            modal = 'text'
        
        if "annotation" in data_dict and data_dict["annotation"] is not None: # grounding data
            if isinstance(data_dict["annotation"], str):
                data_dict["annotation"] = json.loads(data_dict["annotation"])
            conversation = []
            annotations = data_dict["annotation"]
            random.shuffle(annotations)
            phrase_num = 0
            obj_num = 0
            phrase_list = []
            answer_str = ''
            for i,annotation in enumerate(annotations):
                ann_len = len(annotation["ann"]) if "ann" in annotation and annotation["ann"] is not None else 0
                if ann_len>self.max_seg_nums:
                    continue
                if obj_num+ann_len>MAX_OBJ_NUM or phrase_num+1>MAX_PHRASE_NUM:
                    break
                if self.skip_none and annotation["ann"] is None: # 不训none
                    continue
                if annotation['type']=='phrase':
                    phrase = annotation['text'].strip()
                    phrase = phrase.lower()[0]+phrase[1:]
                    if phrase.endswith('.'):
                        phrase = phrase[:-1]  
                    phrase_list.append(phrase)
                    
                else:
                    raise ValueError(f"No phrase or sentence in the annotation")


                mask_ids_inner = []
                if annotation["ann_type"]=="mask": #mask
                    if annotation["ann"] is None:
                        if modal=='image':
                            answer_str += f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}, "
                            phrase_str += f"{REF_START_TOKEN}{phrase}"
                            phrase_num +=1
                            mask_cur = np.zeros((1, h, w)) #[1,h,w]
                            mask_ids.append([len(masks)])
                            masks.append(mask_cur)
                        else:
                            raise ValueError("Null masks are not supported for video samples yet")
                        continue
                    else:
                        mask_valid = True
                        answer_str += f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}, "
                        phrase_str += f"{REF_START_TOKEN}{phrase}"
                        phrase_num +=1
                        obj_num += ann_len

                    # if 'segmentation_file' in data_dict:
                    #     mask_json = json.load(open(os.path.join(data_folder, data_dict['segmentation_file'])))
                    for msk in annotation["ann"]:
                        if isinstance(msk, str):
                            msk = json.loads(msk)
                        if modal=='image':
                            mask_cur = np.expand_dims(annToMask(msk, h, w), axis=0) #[1,h,w]
                        else:
                            mask_cur = np.expand_dims(
                                self._select_video_mask_frame(
                                    msk,
                                    data_dict,
                                    video_frame_candidates,
                                    fallback_image=sam_frame_source,
                                    strict_match=strict_video_mask_match,
                                ),
                                axis=0,
                            )

                        mask_ids_inner.append(len(masks))
                        masks.append(mask_cur)
                    mask_ids.append(mask_ids_inner)
                        
                elif annotation["ann_type"]=="point":
                    mask_type = 2
                    pass
                elif annotation["ann_type"]=="bbox":
                    mask_type = 1
                    answer_str += f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}, "
                    phrase_str += f"{REF_START_TOKEN}{phrase}"
                    phrase_num +=1
                    obj_num += ann_len
                    for bbox in annotation["ann"]:
                        if isinstance(bbox, str):
                            bbox = json.loads(bbox)
                        if modal=='image':
                            mask = np.zeros((h, w))
                            x1, y1, wb, hb = bbox
                            x1 = int(max(x1,0))
                            y1 = int(max(y1,0))
                            wb = int(min(wb, w - x1))
                            hb = int(min(hb, h - y1))
                            mask[y1:y1+hb, x1:x1+wb] = 1
                            mask = np.expand_dims(mask, axis=0) #[1,h,w]
                        else:
                            raise ValueError("Bounding box annotations are not supported for video samples yet")
                        mask_ids_inner.append(len(masks))
                        masks.append(mask)
                    mask_ids.append(mask_ids_inner)

                # if obj_num>=MAX_OBJ_NUM or phrase_num>=MAX_PHRASE_NUM:
                #     break
        
        phrase_str_all = ', '.join(phrase_list)
        # 最后一个换成 and
        if len(phrase_list)>=2:
            last_comma_idx = phrase_str_all.rfind(',')
            phrase_str_all = phrase_str_all[:last_comma_idx] + ' and' + phrase_str_all[last_comma_idx+1:]
        if modal=='image':
            new_contents.append({'type': 'text', 'text': random.choice(SEG_IMAGE_QUESTIONS_PHRASE).format(phrase=phrase_str_all)})
        else:   
            new_contents.append({'type': 'text', 'text': random.choice(SEG_VIDEO_QUESTIONS_PHRASE).format(phrase=phrase_str_all)})
        
        new_conversation.append({"role": "user", "content": new_contents})
        new_conversation.append({"role": "assistant", "content": [
            {'type': 'text', 'text': answer_str.strip().rstrip(',').strip()} # +'.'
        ]})
        # print(new_conversation)

        if len(masks)>0:
            masks = np.array(masks)
            masks = resize_nearest_like_torch(masks, self.mask_size, self.mask_size)
            masks = torch.from_numpy(masks)
        else:
            masks = None

        if 'conversations' in data_dict:
            for idx, conv in enumerate(data_dict['conversations']):
                new_contents.append({'type': 'text', 'text': conv['value']})
                if idx%2==0:
                    message = {"role": "user", "content": new_contents}
                else:
                    message = {"role": "assistant", "content": new_contents}
                new_conversation.append(message)
                new_contents = []
                phrase_str+=conv['value']

        sam_images, sam_size = self._build_sam_image(sam_frame_source)

        # if not mask_valid:
        # print(new_conversation)
        # print('**************')
        # print(phrase_str)
        # print('==============')

        return new_conversation, sam_images, sam_size, masks, mask_ids, mask_valid, mask_type, phrase_str


    def _preprocess(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self._is_compact_revos_train_sample(data_dict):
            data_dict = self._resolve_compact_revos_train_sample(data_dict)

        if data_dict["type"]=="instseg":
            if random.random()>0.5:
                conversation, sam_images, sam_size, masks, mask_ids, mask_valid, mask_type, phrase_str = self._convert_conversation_instseg(data_dict)
            else:
                conversation, sam_images, sam_size, masks, mask_ids, mask_valid, mask_type, phrase_str = self._convert_conversation(data_dict)
        else:
            conversation, sam_images, sam_size, masks, mask_ids, mask_valid, mask_type, phrase_str = self._convert_conversation(data_dict)

        model_inputs = self.processor.apply_chat_template(
            conversation=conversation,
            mm_max_length=self.mm_max_length,
            fps=self.fps,
            max_frames=self.max_frames,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            return_labels=True,
        )

        phrase_ids = self.processor.tokenizer.encode(phrase_str)
        model_inputs["phrase_ids"] = phrase_ids
        model_inputs["sam_images"] = sam_images
        model_inputs["sam_size"] = sam_size
        model_inputs["masks"] = masks
        model_inputs["mask_ids"] = mask_ids
        model_inputs["mask_valid"] = mask_valid
        model_inputs["mask_type"] = mask_type
        if data_dict.get("video") is not None:
            model_inputs["sample_modality"] = "video"
        elif data_dict.get("image") is not None:
            model_inputs["sample_modality"] = "image"
        else:
            model_inputs["sample_modality"] = "text"

        # print(self.processor.decode(model_inputs["input_ids"][0], skip_special_tokens=True))
        # for token_id, label in zip(model_inputs["input_ids"][0], model_inputs["labels"][0]):
        #     token = self.processor.decode([token_id])
        #     if token == "<|image_pad|>":
        #         continue
        #     print([token_id, token, label])
        # exit()

        assert model_inputs["input_ids"].size(-1) <= self.model_max_length, (
            f"Sequence length ({model_inputs['input_ids'].size(-1)}) exceeds model max length ({self.model_max_length})"
        )

        model_inputs["position_ids"] = _get_rope_index(
            model_config=self.model_config,
            **model_inputs,
        )

        return model_inputs

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        image_path = self._dataset[index].get('image', None)
        if image_path is not None and 'objects365_v2_01808559.jpg' in image_path:
            backup_idx = random.randint(0, len(self) - 1)
            print(f"Encounted error when process {index}-th example, use {backup_idx}-th example instead!!!")
            return self.__getitem__(backup_idx)
        try:
            data_dict = self._preprocess(self._dataset[index])
            data_dict["data_index"] = index
        except Exception:
            traceback.print_exc()
            backup_idx = random.randint(0, len(self) - 1)
            print(f"Encounted error when process {index}-th example, use {backup_idx}-th example instead!!!")
            return self.__getitem__(backup_idx)
        return data_dict

    def __len__(self):
        return len(self._dataset)

    def __repr__(self):
        return self._dataset.__repr__()
