#!/bin/bash
SCRIPT=${1:-""}

if [[ -n $DEBUG && $DEBUG -eq 1 ]]; then
    WORLD_SIZE=1
    NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=16667
    RANK=0
fi

if [[ ! -n $NPROC_PER_NODE ]]; then
    NPROC_PER_NODE=$gpu_per_pod
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

set -x

if [[ -d "/data/oss_bucket_0" ]]; then
    mkdir /public && ln -s /data/oss_bucket_0 /public/hz_oss
    mount -o size=100G -o nr_inodes=1000000 -o noatime,nodiratime -o remount /dev/shm
fi

source $SCRIPT
