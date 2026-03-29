import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as maskUtils

def singleMask2rle(mask):
    if mask is None:
        return None
    rle = maskUtils.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

base_dir = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data'

data = json.load(open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/roborefit.json'))

for d in tqdm(data):
    for annotation in d['annotation']:
        new_masks = []
        for msk in annotation['mask']:
            mask_path = os.path.join(base_dir, msk)
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            mask = (mask > 127).astype(np.uint8)  # ensure binary
            # print(mask.shape)
            mask_rle = singleMask2rle(mask)
            new_masks.append(mask_rle)
        annotation['mask'] = new_masks

with open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/roborefit_mask.json', 'w') as f:
    json.dump(data, f, indent=4)
