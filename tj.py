import argparse
import torch

def load_bin_state_dict(path, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)

    # 常见情况：直接就是 state_dict (dict[str, Tensor])
    if isinstance(obj, dict):
        # 有的保存成 {"state_dict": ...} 或 {"model": ...}
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]

        # 尝试认为它本身就是 state_dict
        # 过滤掉明显不是tensor的项（有些 checkpoint 会混杂额外信息）
        sd = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if sd:
            return sd

    raise ValueError(f"Unrecognized .bin format: {path}. Got type={type(obj)}")

def tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a, b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="/mnt/workspace/workgroup/yuanyq/code/video_seg/EasyVLM/work_dirs/qwen3_vl/test/checkpoint-50/non_lora_trainables.bin")
    ap.add_argument("--b", default="/mnt/workspace/workgroup/yuanyq/code/video_seg/EasyVLM/work_dirs/qwen3_vl/test/checkpoint-100/non_lora_trainables.bin")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (used as map_location)")
    args = ap.parse_args()

    sd_a = load_bin_state_dict(args.a, map_location=args.device)
    sd_b = load_bin_state_dict(args.b, map_location=args.device)

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    same = []
    diff = []
    diff_reason = {}

    for k in common:
        ta, tb = sd_a[k], sd_b[k]
        if not (torch.is_tensor(ta) and torch.is_tensor(tb)):
            diff.append(k)
            diff_reason[k] = f"non-tensor entry: {type(ta)} vs {type(tb)}"
            continue

        if ta.dtype != tb.dtype or ta.shape != tb.shape:
            diff.append(k)
            diff_reason[k] = f"shape/dtype differ: {tuple(ta.shape)}/{ta.dtype} vs {tuple(tb.shape)}/{tb.dtype}"
        else:
            if tensor_equal(ta, tb):
                same.append(k)
            else:
                diff.append(k)
                if ta.is_floating_point():
                    max_abs = (ta - tb).abs().max().item()
                    diff_reason[k] = f"same shape/dtype but values differ (max_abs_diff={max_abs})"
                else:
                    diff_reason[k] = "same shape/dtype but values differ"

    print("==== Summary ====")
    print(f"File A: {args.a}")
    print(f"File B: {args.b}")
    print(f"Total keys A: {len(keys_a)}")
    print(f"Total keys B: {len(keys_b)}")
    print(f"Common keys : {len(common)}")
    print(f"Exactly same: {len(same)}")
    print(f"Different   : {len(diff)}")
    print(f"Only in A   : {len(only_a)}")
    print(f"Only in B   : {len(only_b)}")

    same_list = []
    print("\n==== Exactly same tensors (name) ====")
    for k in same:
        if 'language_model' not in k and 'grounding_model.model.vision_encoder.backbone' not in k:
            same_list.append(k)
    diff_list = []
    print("\n==== Same name but different ====")
    for k in diff:
        diff_list.append(k)

    import pdb 
    pdb.set_trace()

    # print("\n==== Only in A ====")
    # for k in only_a:
    #     print(k)

    # print("\n==== Only in B ====")
    # for k in only_b:
    #     print(k)

if __name__ == "__main__":
    main()
