import argparse
import json
from pathlib import Path


def load_payload(path):
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    if path.suffix == ".jsonl":
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError(f"Unsupported result format: {path}")


def strip_summary(payload):
    if (
        payload
        and isinstance(payload, list)
        and isinstance(payload[0], dict)
        and {"j", "f", "j&f"}.issubset(payload[0].keys())
        and "idx" not in payload[0]
    ):
        return payload[1:]
    return payload


def save_payload(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        with open(path, "w") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        return
    if path.suffix == ".jsonl":
        with open(path, "w") as f:
            for item in payload:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return
    raise ValueError(f"Unsupported output format: {path}")


def compute_metrics(results):
    if not results:
        return {"j": 0.0, "f": 0.0, "j&f": 0.0}
    j = sum(item["j"] for item in results) / len(results)
    f = sum(item["f"] for item in results) / len(results)
    return {"j": j, "f": f, "j&f": (j + f) / 2}


def build_progress_payload(results, errors, progress_files):
    total = len(results)
    processed = total
    completed = True
    for progress_file in progress_files:
        if not progress_file.exists():
            completed = False
            continue
        with open(progress_file, "r") as f:
            progress = json.load(f)
        completed = completed and progress.get("completed", False)
    return {
        "processed": processed,
        "total": total,
        "completed": completed,
        "num_errors": len(errors),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="chunk_*.json")
    return parser.parse_args()


def main():
    args = parse_args()
    chunk_dir = Path(args.chunk_dir)
    output_path = Path(args.output_file)

    chunk_files = sorted(chunk_dir.glob(args.pattern))
    if not chunk_files:
        raise FileNotFoundError(
            f"No chunk result files matched {args.pattern} under {chunk_dir}"
        )

    merged_results = []
    merged_errors = []
    progress_files = []

    for chunk_file in chunk_files:
        merged_results.extend(strip_summary(load_payload(chunk_file)))

        error_file = chunk_file.with_name(f"{chunk_file.stem}_errors.json")
        if error_file.exists():
            with open(error_file, "r") as f:
                merged_errors.extend(json.load(f))

        progress_files.append(chunk_file.with_name(f"{chunk_file.stem}_progress.json"))

    merged_results.sort(key=lambda item: item.get("idx", 0))
    metrics = compute_metrics(merged_results)
    save_payload([metrics] + merged_results, output_path)

    error_output = Path(str(output_path.with_suffix("")) + "_errors.json")
    if merged_errors:
        save_payload(merged_errors, error_output)

    progress_output = Path(str(output_path.with_suffix("")) + "_progress.json")
    progress_payload = build_progress_payload(merged_results, merged_errors, progress_files)
    progress_payload["results_saved_to"] = str(output_path)
    progress_payload["errors_saved_to"] = str(error_output) if merged_errors else None
    save_payload(progress_payload, progress_output)

    print(
        f"Merged {len(chunk_files)} chunk files, {len(merged_results)} samples, "
        f"{len(merged_errors)} errors -> {output_path}"
    )
    print(
        f"J={metrics['j']:.4f} F={metrics['f']:.4f} J&F={metrics['j&f']:.4f}"
    )


if __name__ == "__main__":
    main()
