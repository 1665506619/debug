import argparse
import os
import random
import string
from functools import partial

import shutil
import gradio as gr
from datasets import load_dataset


TMP_DIR = os.path.join("/tmp", "data_browser")
os.makedirs(TMP_DIR, exist_ok=True)


def tmp_path(path):
    characters = string.ascii_letters + string.digits
    name = "".join(random.choices(characters, k=16))
    target_path = os.path.join(TMP_DIR, name + os.path.splitext(path)[-1])
    shutil.copy(path, target_path)
    return target_path


def on_index_change(index, selected_ann, datasets):
    ann_index, _ = selected_ann
    item = datasets[ann_index][index]
    messages = []
    for message in item["conversation"]:
        role = message["role"]
        for content in message["content"]:
            if content["type"] == "text":
                content = content["text"]
            elif content["type"] == "image":
                content = gr.Image(tmp_path(content["image"]))
            elif content["type"] == "video":
                if os.path.isfile(content["video"]):
                    content = gr.Video(tmp_path(content["video"]))
                else:
                    content = content["video"]
            messages.append(gr.ChatMessage(role=role, content=content))
    return messages


def on_change_annotation(selected_ann, datasets):
    ann_index, (ann_name, _, _) = selected_ann
    dataset = datasets[ann_index]
    messages = on_index_change(0, selected_ann, datasets)
    return ann_name, gr.update(maximum=len(dataset) - 1, value=0, interactive=True), messages


def on_load(ann_dir, cache_dir):
    ann_files = [x for x in os.listdir(ann_dir) if x.endswith(".jsonl")]

    ann_names, datasets = [], []
    for i, ann_file in enumerate(ann_files):
        ann_name = os.path.splitext(ann_file)[0]
        gr.Info(f"Loading {i + 1}/{len(ann_files)} dataset: {ann_name}", duration=2)
        ann_path = os.path.join(ann_dir, ann_file)
        dataset = load_dataset("json", data_files=ann_path, cache_dir=cache_dir)["train"]
        ann_names.append(ann_name)
        datasets.append(dataset)

    num_items = [len(dataset) for dataset in datasets]
    total_items = sum(num_items)

    indices = sorted(
        list(range(len(ann_files))),
        key=lambda x: num_items[x],
        reverse=True,
    )

    percentages = [f"{n / total_items * 100:.2f}%" for n in num_items]
    num_items = [f"{n:,d}" for n in num_items]
    ann_infos = list(zip(ann_names, num_items, percentages))

    datasets = [datasets[i] for i in indices]
    ann_infos = [ann_infos[i] for i in indices]

    return datasets, gr.Dataset(samples=ann_infos)


def main():
    parser = argparse.ArgumentParser(description="Data Browser Tool")
    parser.add_argument("ann_dir", help="Path to the annotation directory")
    parser.add_argument("--cache-dir", "--cache_dir", type=str, default=None, help="Path to the cache directory")
    args = parser.parse_args()

    with gr.Blocks() as demo:
        datasets = gr.State()

        with gr.Row():
            selected = gr.Text(label="Selected Dataset", interactive=False)
            index = gr.Slider(minimum=0, maximum=1, step=1, label="Index", interactive=False)

        with gr.Row():
            annotations = gr.Dataset(
                components=[gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False)],
                headers=["Name", "Num", "Percentage"],
                type="tuple",
                label="Annotations",
                scale=1,
            )
            chatbot = gr.Chatbot(type="messages")

        demo.load(
            fn=partial(on_load, ann_dir=args.ann_dir, cache_dir=args.cache_dir),
            inputs=[],
            outputs=[datasets, annotations],
        )

        annotations.click(
            fn=on_change_annotation,
            inputs=[annotations, datasets],
            outputs=[selected, index, chatbot],
        )

        index.change(
            fn=on_index_change,
            inputs=[index, annotations, datasets],
            outputs=[chatbot],
        )

    demo.launch(server_name="0.0.0.0", ssl_verify=False)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP_DIR)
