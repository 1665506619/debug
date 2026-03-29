# torchrun --nnodes 1 --nproc_per_node 4 -m tests.encoder_load_balancing --seed 123 --size 2
import argparse
import time
from functools import partial

import torch
from tqdm import tqdm
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionConfig
from transformers.trainer_utils import set_seed

import easy_vlm.training.utils as utils
from easy_vlm.models.qwen3_vl import Qwen3VLVisionModel


class TrainingArguments(utils.TrainingArguments):
    def __post_init__(self):
        return


def build_model():
    config = Qwen3VLVisionConfig(spatial_merge_size=2)
    model = Qwen3VLVisionModel._from_config(
        config,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.cuda()
    model.gradient_checkpointing_enable()

    model = torch.nn.parallel.DistributedDataParallel(model)

    return model


def build_inputs(model: Qwen3VLVisionModel):
    num_items = torch.randint(1, 16, size=(1,), device="cuda")

    grid_thw = (
        torch.randint(
            32 // 16,
            256 // 16 + 1,
            (num_items, 3),
            device="cuda",
        )
        * 2
    )
    grid_thw[:, 0] = torch.randint(1, 32 + 1, size=(num_items,), device="cuda")

    num_tokens = grid_thw.prod(dim=1).sum()
    model = model.module
    pixel_values = torch.randn(
        (
            num_tokens,
            model.patch_size**2 * model.config.in_channels * model.config.temporal_patch_size,
        ),
        dtype=torch.bfloat16,
        device="cuda",
    )

    return {
        "hidden_states": pixel_values,
        "grid_thw": grid_thw,
    }


def rank0_print(*args, **kwargs):
    if torch.distributed.get_rank() == 1:
        print(*args, **kwargs)


def forward(model, inputs=None, balance=False):
    utils._ARGS.encoder_load_balancing = balance

    if inputs is None:
        inputs = build_inputs(model)

    torch.distributed.barrier()
    torch.cuda.synchronize()
    start_time = time.time()

    outputs = model(**inputs)

    torch.distributed.barrier()
    torch.cuda.synchronize()
    t = (time.time() - start_time) * 1000

    return outputs, t


def test_correctness(model):
    inputs = build_inputs(model)

    outputs, _ = forward(model, inputs, balance=False)
    outputs_balance, _ = forward(model, inputs, balance=True)

    rank0_print(outputs[0])
    rank0_print(outputs_balance[0])

    assert torch.allclose(outputs[0], outputs_balance[0])
    for tensor_1, tensor_2 in zip(outputs[1], outputs_balance[1]):
        assert torch.allclose(tensor_1, tensor_2)

    rank0_print("✅ Correctnesss test passed!")


@torch.inference_mode()
def record_time(func, iterations=20):
    for _ in range(10):
        func()

    times = []
    for _ in tqdm(range(iterations)):
        _, t = func()
        times.append(t)

    return sum(times) / len(times)


def test_time(model):
    inputs = build_inputs(model)
    time_no_balance = record_time(partial(forward, model=model, inputs=inputs, balance=False))
    rank0_print(f"Time without balancing: {time_no_balance:.2f} ms")
    torch.distributed.barrier()

    time_balance = record_time(partial(forward, model=model, inputs=inputs, balance=True))
    rank0_print(f"Time with balancing: {time_balance:.2f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--size", type=int, default=2)
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(torch.distributed.get_rank())

    training_args = TrainingArguments()
    utils._ARGS = training_args

    rank = torch.distributed.get_rank()
    ranks = torch.arange(torch.distributed.get_world_size())
    groups = ranks.view(-1, args.size)

    for group in groups:
        pg = torch.distributed.new_group(
            ranks=group.tolist(),
            backend="nccl",
        )
        if rank in group:
            rank0_print(pg, group)
            utils._ENCODER_LOAD_BALANCING_GROUP = pg

    set_seed(seed=args.seed + torch.distributed.get_rank(), deterministic=True)
    model = build_model()

    test_correctness(model)
    test_time(model)


if __name__ == "__main__":
    main()
