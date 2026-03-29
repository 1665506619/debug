import os

from transformers import AutoConfig
from .sam3 import Sam3Config, Sam3Model
import torch.nn as nn


def _set_num_queries(config, num_queries):
    if hasattr(config, "detector_config") and hasattr(config.detector_config, "detr_decoder_config"):
        config.detector_config.detr_decoder_config.num_queries = num_queries
        return
    if hasattr(config, "detr_decoder_config"):
        config.detr_decoder_config.num_queries = num_queries
        return
    raise AttributeError("Unsupported SAM3 config structure: missing detr decoder config")


class SegmentationDecoder(nn.Module):
    def __init__(self, config):
        super(SegmentationDecoder, self).__init__()
        self.config = config
        if config.seg_encoder=='sam3' and config.seg_decoder=='sam3':
            config = self._load_sam3_config()
            _set_num_queries(config, self.config.max_seg_nums)
        
            self.model = Sam3Model(config)
            # self.mask_encoder = self.model.vision_encoder
            # self.mask_decoder = self.model
        else:
            raise NotImplementedError

    def _load_sam3_config(self):
        mask_decoder_model = getattr(self.config, "mask_decoder_model", None)
        if mask_decoder_model:
            try:
                return AutoConfig.from_pretrained(mask_decoder_model)
            except Exception as exc:
                print(
                    "Warning: failed to load SAM3 config from "
                    f"{mask_decoder_model}: {exc}. Falling back to default Sam3Config."
                )
        return Sam3Config()

    def load_model(self, config):
        mask_decoder_model = getattr(self.config, "mask_decoder_model", None)
        original_config = self._load_sam3_config()
        _set_num_queries(original_config, config.max_seg_nums)
        if mask_decoder_model and os.path.exists(mask_decoder_model):
            self.model = Sam3Model.from_pretrained(
                mask_decoder_model,
                config=original_config,
                ignore_mismatched_sizes=True,
            )
        else:
            print(
                "Warning: SAM3 mask decoder weights directory is unavailable; "
                "using randomly initialized decoder weights."
            )
            self.model = Sam3Model(original_config)

    def encoder(self, pixel_values):
        vision_outputs = self.model.vision_encoder(pixel_values)
        return vision_outputs


    def decoder(self, vision_outputs, text_embeds, text_attn_mask, query_embed):
        mask_outputs = self.model(
            vision_embeds = vision_outputs,
            attention_mask=text_attn_mask,
            text_embeds = text_embeds,
            query_embed = query_embed,
        )
        return mask_outputs
