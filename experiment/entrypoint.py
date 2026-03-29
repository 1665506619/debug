import os
import argparse
import subprocess

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str)
    args = parser.parse_args()

    env = os.environ.copy()

    if os.environ.get("DEBUG", "0") == "1":
        env["WORLD_SIZE"] = "1"
        env["NPROC_PER_NODE"] = str(torch.cuda.device_count())
        env["RANK"] = "0"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = "16666"
    else:
        env["WORLD_SIZE"] = os.environ.get("WORLD_SIZE")
        env["NPROC_PER_NODE"] = os.environ.get("NPROC_PER_NODE", os.environ.get("gpu_per_pod"))
        env["RANK"] = os.environ.get("RANK")
        env["MASTER_ADDR"] = os.environ.get("MASTER_ADDR")
        env["MASTER_PORT"] = os.environ.get("MASTER_PORT")

    os.environ.update(env)

    if os.path.exists("/data/oss_bucket_0"):
        os.makedirs("/public", exist_ok=True)
        os.system(f"ln -s /data/oss_bucket_0 /public/hz_oss")
        os.system("mount -o size=100G -o nr_inodes=1000000 -o noatime,nodiratime -o remount /dev/shm")

    os.system("apt-get update && apt-get install -y ffmpeg")
    os.system("yum makecache && yum install -y ffmpeg")

    print(f"CMD: bash {args.script}")
    print(f"ENV: {os.environ}")

    subprocess.run(args=["bash", args.script])


if __name__ == "__main__":
    main()
