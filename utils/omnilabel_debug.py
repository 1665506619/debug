import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from omnilabeltools import OmniLabel  # 你也可以换成你封装的 snake_case OmniLabel


def clean_name_like(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    # 合并连续空白
    s = " ".join(s.split())
    return s


class OmniLabelTorchDataset(Dataset):
    def __init__(
        self,
        ann_file: str,
        img_prefix: str = "",
        skip_catg: bool = False,
        load_image: bool = False,
        image_loader: Optional[Callable[[str], torch.Tensor]] = None,
    ):
        super().__init__()
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.skip_catg = skip_catg
        self.load_image = load_image
        self.image_loader = image_loader

        self.ol = OmniLabel(path_json=ann_file)
        self.img_ids = list(self.ol.image_ids)

        self.data_list = self._build_data_list()

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.data_list[idx]

        if self.load_image:
            img_path = data["img_path"]
            if self.image_loader is not None:
                image = self.image_loader(img_path)
            else:
                image = self._default_load_image_as_tensor(img_path)
            out = dict(data)
            out["image"] = image
            return out

        return data

    def _build_data_list(self) -> List[Dict[str, Any]]:
        data_list: List[Dict[str, Any]] = []

        for img_id in self.img_ids:
            img_info = self.ol.get_image_sample(img_id)
            ann_info = img_info.get("instances", [])
            file_name = img_info['file_name']
            img_path = osp.join(self.img_prefix, file_name) if self.img_prefix else file_name

            # 处理 instances（bbox: xywh -> xyxy；过滤 ignore/非法框）
            instances: List[Dict[str, Any]] = []
            for ann in ann_info:
                if ann.get("ignore", False):
                    continue

                x1, y1, w, h = ann["bbox"]
                area = ann.get("area", w * h)

                if area <= 0 or w < 1 or h < 1:
                    continue

                bbox_xyxy = [x1, y1, x1 + w, y1 + h]
                ignore_flag = 1 if ann.get("iscrowd", False) else 0

                instances.append(
                    {
                        "bbox": bbox_xyxy,
                        "ignore_flag": ignore_flag,
                    }
                )

            text: List[str] = []
            description_ids: List[int] = []

            for ls in img_info.get("labelspace", []):
                if not self.skip_catg:
                    text.append(clean_name_like(ls["text"]))
                    description_ids.append(ls["id"])
                else:
                    # 你原逻辑：skip_catg=True 时，跳过 type == "C"
                    if ls.get("type") != "C":
                        text.append(clean_name_like(ls["text"]))
                        description_ids.append(ls["id"])

            data_info: Dict[str, Any] = {
                "img_path": img_path,
                "img_id": img_info["id"],
                "custom_entities": False,
                "tokens_positive": -1,
                "text": text,
                "description_ids": description_ids,
                "instances": instances,
            }
            data_list.append(data_info)

        return data_list

    @staticmethod
    def _default_load_image_as_tensor(img_path: str) -> torch.Tensor:
        from PIL import Image
        import numpy as np

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            arr = np.array(im)  # H,W,3 uint8
        # -> float32 tensor C,H,W
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
        return t

if __name__ == "__main__":
    omnilabel = OmniLabelTorchDataset(
        ann_file='/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/eval/omnilabel_val_v0.1.3.json',
        img_prefix='/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/omnilabel_images'
    )
    import pdb 
    pdb.set_trace()