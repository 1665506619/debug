import argparse
import os
import subprocess


NAS_CONFIGS = [
    {
        "nas_file_system_mount_path": "/mnt/damovl",
        "nas_file_system_id": "9465b4bec1-drd63.cn-zhangjiakou.nas.aliyuncs.com",
    },
    {
        "nas_file_system_mount_path": "/mnt/damorobot",
        "nas_file_system_id": "9bdd1490ea-oyk11.cn-zhangjiakou.nas.aliyuncs.com",
    },
    {
        "nas_file_system_mount_path": "/mnt/workspace/workgroup",
        "nas_file_system_id": "9281a4908f-tva22.cn-zhangjiakou.nas.aliyuncs.com",
    },
    {
        "nas_file_system_mount_path": "/mnt/rynnbot_hangzhou",
        "nas_file_system_id": "0b1084992d-ifc98.cn-hangzhou.nas.aliyuncs.com",
    },
]

OSS_CONFIGS = [
    {

        "oss_bucket": "damo-xlab-hangzhou",
        "oss_endpoint": "cn-hangzhou.oss.aliyuncs.com",
    },
]

DEFAULT_CONFIG = {
    "queue": "damo_eai_rfm_l20x",
    "nebula_work_dir": os.path.dirname(os.path.dirname(__file__)),
    "file.cluster_file": os.path.join("experiment", "configs", "node_config.json"),
}

if len(NAS_CONFIGS):
    DEFAULT_CONFIG.update({k: ",".join([config[k] for config in NAS_CONFIGS]) for k in NAS_CONFIGS[0]})

if len(OSS_CONFIGS):
    DEFAULT_CONFIG.update({k: ",".join([config[k] for config in OSS_CONFIGS]) for k in OSS_CONFIGS[0]})

ENTRYPOINT = os.path.join(os.path.basename(os.path.dirname(__file__)), "entrypoint.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--custom-image", "--custom_image", action="store_true")
    args = parser.parse_args()

    if args.debug:
        os.environ["DEBUG"] = "1"
        command = ["python", ENTRYPOINT, args.script]
    else:
        command = ["nebulactl", "run", "mdl"]
        command += [f"--job_name={os.path.splitext(os.path.basename(args.script))[0]}"]
        command += [f"--worker_count={args.nnodes}"]
        command += [f"--{k}={v}" for k, v in DEFAULT_CONFIG.items()]
        if args.custom_image:
            command += ["--entry=python"]
            command += [f"--user_params={ENTRYPOINT} {args.script}"]
            command += ["--custom_docker_image=hub.docker.alibaba-inc.com/aone-mlflow/custom-image-damo-embodied-intelligence:ngc-pytorch-25.04-py3-train"]
        else:
            command += [f"--entry={ENTRYPOINT}"]
            command += [f"--user_params={args.script}"]
            command += ["--algo_name=pytorch280"]

    print(command)
    subprocess.run(command)


if __name__ == "__main__":
    main()
