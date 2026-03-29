#!/usr/bin/env python3
"""
RoboRefIt Dataset Construction Script

Convert RoboRefIt raw data to evaluation format:
[
  {
    "image": "RoboRefIt/testA/image/0000000.png",
    "question": "I want to get the blue cone",
    "masks": {
      "size": [H, W],
      "counts": "..."
    }
  },
  ...
]
"""

import json
import os
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
from tqdm import tqdm
import argparse


def singleMask2rle(mask):
    """Convert binary mask to RLE format"""
    if mask is None:
        return None
    rle = maskUtils.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def load_mask_from_png(mask_path):
    """Load mask from PNG file"""
    if not os.path.exists(mask_path):
        print(f"Warning: Mask file not found: {mask_path}")
        return None
    
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)
    # Binary mask: non-zero pixels are foreground
    mask = (mask > 0).astype(np.uint8)
    return mask


def construct_roborefit(data_root, split='testA', output_dir='data/eval'):
    """
    Construct RoboRefIt evaluation data
    
    Args:
        data_root: Root directory of RoboRefIt dataset
        split: 'testA' or 'testB'
        output_dir: Output directory for processed JSON
    """
    print(f"{'='*60}")
    print(f"Constructing RoboRefIt {split} dataset")
    print(f"{'='*60}")
    
    # Paths
    split_dir = os.path.join(data_root, split)
    json_file = os.path.join(split_dir, f'roborefit_{split}.json')
    image_dir = os.path.join(split_dir, 'image')
    mask_dir = os.path.join(split_dir, 'mask')
    
    # Check if files exist
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    
    # Load raw data
    print(f"Loading raw data from: {json_file}")
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    
    print(f"Total samples: {len(raw_data)}")
    
    # Process data
    final_data = []
    skipped = 0
    
    for item in tqdm(raw_data, desc=f"Processing {split}"):
        try:
            # Extract fields
            text = item['text']
            rgb_path = item['rgb_path']
            mask_path = item['mask_path']
            
            # Fix Windows path separators
            rgb_path = rgb_path.replace('\\', '/')
            mask_path = mask_path.replace('\\', '/')
            
            # Extract image filename from path
            # e.g., "final_dataset/testA/image/0000000.png" -> "0000000.png"
            image_filename = os.path.basename(rgb_path)
            
            # Extract mask folder and filename
            # e.g., "final_dataset/testA/mask/0000000/01.png" -> "0000000/01.png"
            mask_parts = mask_path.split('/')
            mask_folder = mask_parts[-2]  # e.g., "0000000"
            mask_filename = mask_parts[-1]  # e.g., "01.png"
            
            # Construct actual paths
            actual_image_path = os.path.join(image_dir, image_filename)
            actual_mask_path = os.path.join(mask_dir, mask_folder, mask_filename)
            
            # Load mask
            mask = load_mask_from_png(actual_mask_path)
            if mask is None:
                skipped += 1
                continue
            
            # Convert mask to RLE
            mask_rle = singleMask2rle(mask)
            if mask_rle is None:
                skipped += 1
                continue
            
            # Construct relative path for image (relative to data root)
            relative_image_path = f"RoboRefIt/{split}/image/{image_filename}"
            
            # Create data entry
            data_entry = {
                "image": relative_image_path,
                "question": text,
                "masks": mask_rle
            }
            
            final_data.append(data_entry)
            
        except Exception as e:
            print(f"\nError processing item {item.get('num', '?')}: {e}")
            skipped += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total samples: {len(raw_data)}")
    print(f"Valid samples: {len(final_data)}")
    print(f"Skipped samples: {skipped}")
    print(f"{'='*60}\n")
    
    # Save to output file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'roborefit_{split}_qa.json')
    
    print(f"Saving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"✅ Successfully saved {len(final_data)} samples to {output_file}")
    
    # Print sample
    if len(final_data) > 0:
        print(f"\n{'='*60}")
        print("Sample entry:")
        print(f"{'='*60}")
        sample = final_data[0]
        print(f"Image: {sample['image']}")
        print(f"Question: {sample['question']}")
        print(f"Mask size: {sample['masks']['size']}")
        print(f"Mask counts (first 50 chars): {sample['masks']['counts'][:50]}...")
        print(f"{'='*60}\n")
    
    return final_data


def main():
    parser = argparse.ArgumentParser(description='Construct RoboRefIt evaluation data')
    parser.add_argument(
        '--data_root',
        type=str,
        default='/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/RoboRefIt',
        help='Root directory of RoboRefIt dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval',
        help='Output directory for processed JSON files'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['testA', 'testB'],
        help='Splits to process (testA, testB, or both)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("RoboRefIt Dataset Construction")
    print(f"{'='*60}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"{'='*60}\n")
    
    # Process each split
    for split in args.splits:
        try:
            construct_roborefit(
                data_root=args.data_root,
                split=split,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"❌ Error processing {split}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("All splits processed!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

