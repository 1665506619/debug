"""
Format raw annotation files to the standard format for training.

Functionality:
    - Convert LLaVA-style conversations to ChatML format.
    - Get image and video metadata (width, height, duration) if missing.
    - Format `messages` and `contents` so that they have consistent structure across datasets.
    - Calculating and adding text sequence length.

Arguments:
    --ann_path: Path to the raw annotation file(s). Supported format: `json`, `jsonl`, and `parquet`.
    --save_path: (Optional) Path to save the formatted annotation file (jsonl).
        Can be `None` for checking and debugging mode.
    --tokenizer_path: (Optional) Path to the tokenizer for text sequence length calculation.
        For example, `Qwen/Qwen3-VL-2B-Instruct`.
    --cache_dir: (Optional) Cache directory for HF dataset.
    --data_root: (Optional) Root directory to resolve relative paths in annotations.
    --nproc: (Optional) Number of parallel processes to use, default is the number of CPU cores.
    --yes/-y: Overwrite existing files without prompt.
    --check: Only check the data without formatting.
    --debug: Enable debug mode which use single process and will not save the formatted annotations.
"""

import argparse
import os
import os.path as osp
import traceback
from multiprocessing import Process, Queue, cpu_count

import ffmpeg
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer


# Configs
DATA_KEYS = {"data_source", "id", "text_sequence_length", "conversation"}
MESSAGE_KEYS = {"role", "content"}
VALID_ROLES = {"system", "user", "assistant"}
VALID_MODALITIES = {"text", "image", "video"}
CONTENT_KEYS = VALID_MODALITIES.union({"type", "width", "height", "duration"})
SORTED_CONTENT_KEYS = ["type", "text", "image", "video", "width", "height", "duration"]


def convert_llava_conversation(conversation, images, videos, widths=None, heights=None):
    if isinstance(images, str):
        images = [images]
    if isinstance(videos, str):
        videos = [videos]

    assert isinstance(images, list), "Images should be a list."
    assert isinstance(videos, list), "Videos should be a list."

    first_user_idx = 0
    for i, message in enumerate(conversation):
        if message["from"] == "human" or message["from"] is None:
            first_user_idx = i
            break

    if len(images) and all("<image>" not in message["value"] for message in conversation):
        conversation[first_user_idx]["value"] = "<image>" + conversation[first_user_idx]["value"]
    if len(videos) and all("<video>" not in message["value"] for message in conversation):
        conversation[first_user_idx]["value"] = "<video>" + conversation[first_user_idx]["value"]

    image_idx, video_idx = 0, 0
    messages = []

    for conv in conversation:
        text = conv["value"]
        contents = []

        while len(text):
            image_pos, video_pos = text.find("<image>"), text.find("<video>")
            if image_pos == -1 and video_pos == -1:
                contents.append({"type": "text", "text": text})
                text = ""

            elif video_pos == -1 or (image_pos != -1 and image_pos < video_pos):
                if len(text[:image_pos]):
                    contents.append({"type": "text", "text": text[:image_pos]})

                if len(images) > 0:
                    content = {"type": "image", "image": images[image_idx]}
                    if widths is not None and heights is not None:
                        content.update({"width": widths[image_idx], "height": heights[image_idx]})
                    contents.append(content)
                    text = text[image_pos + len("<image>") :].lstrip()
                    image_idx += 1
                else:
                    contents.append({"type": "text", "text": "<image>"})
                    text = text[image_pos + len("<image>") :]

            elif image_pos == -1 or (video_pos != -1 and video_pos < image_pos):
                if len(text[:video_pos]):
                    contents.append({"type": "text", "text": text[:video_pos]})

                if len(videos) > 0:
                    contents.append({"type": "video", "video": videos[video_idx]})
                    text = text[video_pos + len("<video>") :].lstrip()
                    video_idx += 1
                else:
                    contents.append({"type": "text", "text": "<video>"})
                    text = text[video_pos + len("<video>") :]

            else:
                raise ValueError("Unknown error")

        if conv["from"] == "human":
            role = "user"
        elif conv["from"] == "gpt":
            role = "assistant"
        elif conv["from"] == "system":
            role = "system"
        else:
            raise ValueError(f"Unknown role: {conv['from']}")
        messages.append({"role": role, "content": contents})

    assert image_idx == len(images), f"Not all images were used in the conversation ({image_idx} vs {len(images)})"
    assert video_idx == len(videos), f"Not all videos were used in the conversation ({video_idx} vs {len(videos)})"

    return messages


def get_image_metadata(image_path):
    image = Image.open(image_path)
    return {"width": image.width, "height": image.height}


def get_video_metadata(video_path):
    if isinstance(video_path, (list, tuple)) or osp.isdir(video_path):
        if isinstance(video_path, (list, tuple)):
            frames = video_path
        else:
            frames = [x for x in os.listdir(video_path) if x.endswith((".png", ".jpg", ".jpeg"))]
        frame = Image.open(osp.join(video_path, frames[0]))
        fps = 2
        return {
            "width": frame.width,
            "height": frame.height,
            "duration": float(len(frames) / fps),
        }

    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"))
    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "duration": float(probe["format"]["duration"]),
    }


def get_text_sequence_length(conversation, tokenizer):
    assert tokenizer is not None, "Tokenizer is required to calculate text sequence length."
    texts = []
    for message in conversation:
        for content in message["content"]:
            if content["type"] == "text":
                texts.append(content["text"])
    prompt = "\n".join(texts)
    input_ids = tokenizer(prompt)["input_ids"]
    return len(input_ids)


def convert_data(data, data_source, index, tokenizer=None, data_root=None):
    if isinstance(data, list):
        # Assume it is conversation
        data = {"conversation": data}

    if not data.get("data_source", ""):
        data["data_source"] = str(data_source)
    if not data.get("id", ""):
        data["id"] = str(index)

    if "conversations" in data:
        # LLaVA format
        data = {
            "conversation": convert_llava_conversation(
                data.pop("conversations"),
                data.get("image", []),
                data.get("video", []),
                data.get("widths", None),
                data.get("heights", None),
            ),
            **data,
        }

    for message in data["conversation"]:
        assert "content" in message, f"Message must have content, but got {message}"

        if isinstance(message["content"], str):
            # Assume it is pure text
            message["content"] = [{"type": "text", "text": message["content"]}]

        new_contents = []
        for content in message["content"]:
            modality = content.get("type", None)
            assert modality is not None, f"Content must has type, but got {content}"

            assert modality in VALID_MODALITIES, f"Invalid modality: {modality}"
            assert content.get(modality, None) is not None, f"Unable to find data in content: {content}"

            mm_data = content[modality]
            if modality == "text":
                if isinstance(mm_data, list) and len(mm_data) == 1:
                    mm_data = mm_data[0]
                    content[modality] = mm_data

            else:
                if isinstance(mm_data, (list, tuple)) and not all([osp.isabs(x) for x in mm_data]):
                    assert data_root is not None, f"data_root is required to resolve relative path: {mm_data}"
                    mm_data = [x if osp.isabs(x) else osp.join(data_root, x) for x in mm_data]
                    content[modality] = mm_data
                elif isinstance(mm_data, str) and not osp.isabs(mm_data):
                    assert data_root is not None, f"data_root is required to resolve relative path: {mm_data}"
                    mm_data = osp.join(data_root, mm_data)
                    content[modality] = mm_data

                width, height = content.get("width", None), content.get("height", None)
                if width == height == 0 or width == height == 1:
                    content.pop("width")
                    content.pop("height")

                if modality == "image" and any([key not in content for key in ["width", "height"]]):
                    assert isinstance(mm_data, str)
                    content.update(get_image_metadata(mm_data))
                elif modality == "video" and any([key not in content for key in ["width", "height", "duration"]]):
                    assert isinstance(mm_data, (str, list, tuple))
                    content.update(get_video_metadata(mm_data))
                    if isinstance(mm_data, (list, tuple)):
                        content[modality] = json.dumps(mm_data, ensure_ascii=False)

            for key in ["text", "image", "video"]:
                if key not in content:
                    content[key] = ""
            for key in ["width", "height"]:
                if key not in content:
                    content[key] = 0
            for key in ["duration"]:
                if key not in content:
                    content[key] = 0.0

            new_contents.append({k: content[k] for k in SORTED_CONTENT_KEYS})

        message["content"] = new_contents

    if "text_sequence_length" not in data:
        data["text_sequence_length"] = get_text_sequence_length(data["conversation"], tokenizer)

    data = {
        "data_source": str(data.get("data_source", "unknown")),
        "id": str(data.get("id", "unknown")),
        "text_sequence_length": data["text_sequence_length"],
        "conversation": data["conversation"],
    }

    return data


def check_data(data):
    assert isinstance(data, dict), "Data must be a dictionary."
    assert set(data.keys()) == DATA_KEYS, f"Invalid data keys: {list(data.keys())}, expected: {list(DATA_KEYS)}"

    for message in data["conversation"]:
        assert isinstance(message, dict), "Message must be a dictionary."
        assert set(message.keys()) == MESSAGE_KEYS, (
            f"Invalid message keys: {list(message.keys())}, expected: {list(MESSAGE_KEYS)}"
        )
        assert message["role"] in VALID_ROLES, f"Invalid role: {message['role']}"
        assert isinstance(message["content"], list), f"Content must be list, but got {type(message['content'])}"

        for content in message["content"]:
            assert isinstance(content, dict), "Content must be a dictionary."
            assert content.keys() == CONTENT_KEYS, (
                f"Invalid content keys: {list(content.keys())}, expected: {list(CONTENT_KEYS)}"
            )
            modality = content.get("type", None)
            assert modality in VALID_MODALITIES, f"Invalid modality: {modality}"
            mm_data = content.get(modality, None)
            assert isinstance(mm_data, str), f"Content data must be string, but got {type(mm_data)}"


def processor(
    input_buffer,
    output_buffer=None,
    tokenizer_path=None,
    should_convert=True,
    data_root=None,
):
    tokenizer = None if tokenizer_path is None else AutoTokenizer.from_pretrained(tokenizer_path)

    while True:
        data = input_buffer.get()
        if isinstance(data, str) and data == "<FINISH>":
            input_buffer.put(data)
            if output_buffer is not None:
                output_buffer.put(data)
            break

        data, data_source, index = data

        if should_convert:
            try:
                data = convert_data(data, data_source, index, tokenizer, data_root)
            except Exception:
                traceback.print_exc()
                print(f"Data conversion failed: {data}, please check the traceback above.")
                continue

        try:
            check_data(data)
        except Exception:
            traceback.print_exc()
            print(f"Data check failed: {data}, please check the traceback above.")
            continue

        if output_buffer is not None:
            output_buffer.put(json.dumps(data, ensure_ascii=False))


def writer(save_path, cache_dir, output_buffer, num_processes):
    num_finish_signals = 0
    with tqdm(desc="Writing", position=1) as pbar:
        with open(save_path, "w") as f:
            while True:
                if num_finish_signals == num_processes:
                    break
                data = output_buffer.get()
                if data == "<FINISH>":
                    num_finish_signals += 1
                    continue
                f.write(data + "\n")
                pbar.update(1)

    print("Creating HF dataset...")
    dataset = load_dataset("json", data_files=save_path, cache_dir=cache_dir)
    print(dataset)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ann_path", type=str, required=True, nargs="+")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--nproc", type=int, default=None)

    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()

    if args.nproc is None:
        if args.debug:
            args.nproc = 1
        else:
            args.nproc = cpu_count()

    input_buffer = Queue(maxsize=args.nproc * 8)

    if not args.check and not args.debug:
        assert args.save_path is not None, "save_path is required unless in check or debug mode."
        os.makedirs(osp.dirname(osp.abspath(args.save_path)), exist_ok=True)

        output_buffer = Queue(maxsize=args.nproc * 8)

        assert args.save_path.endswith(".jsonl"), "Only support saving as .jsonl format."
        if osp.exists(args.save_path) and not args.yes:
            override = input(f"The file `{args.save_path}` already exists, do you want to overwrite it? (y/N) ")
            if override.lower() != "y":
                exit()

        writer_p = Process(
            target=writer, args=(args.save_path, args.cache_dir, output_buffer, args.nproc), daemon=True
        )
        writer_p.start()

    else:
        output_buffer = None
        writer_p = None

    for _ in range(args.nproc):
        p = Process(
            target=processor,
            args=(input_buffer, output_buffer, args.tokenizer_path, not args.check, args.data_root),
            daemon=True,
        )
        p.start()

    for ann_path in args.ann_path:
        if args.save_path:
            default_data_source = osp.splitext(osp.basename(args.save_path))[0]
        else:
            default_data_source = None

        if ann_path.endswith(".json"):
            with open(ann_path, "r") as f:
                data_list = json.load(f)
                for i, data in enumerate(tqdm(data_list, desc="Loading", position=0)):
                    input_buffer.put((data, default_data_source, i))

        elif ann_path.endswith(".jsonl"):
            with open(ann_path, "r") as f:
                with tqdm(desc="Loading", position=0) as pbar:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if len(line) > 0:
                            input_buffer.put((json.loads(line), default_data_source, i))
                        pbar.update(1)

        else:
            dataset = load_dataset("parquet", data_files=ann_path)["train"]
            for i, data in enumerate(tqdm(dataset, desc="Loading", position=0)):
                input_buffer.put((data, default_data_source, i))

    input_buffer.put("<FINISH>")

    if writer_p is not None:
        writer_p.join()


if __name__ == "__main__":
    main()
