#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=8

RESUME=$1
ARCH=vgg16

DATASET=${2-pitts}
SCALE=${3-250k}

if [ $# -lt 1 ]
  then
    echo "Arguments error: <MODEL PATH>"
    echo "Optional arguments: <DATASET (default:pitts)> <SCALE (default:250k)>"
    exit 1
fi

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/test.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
    --test-batch-size 32 -j 2 \
    --vlad --reduction \
    --resume ${RESUME}
    # --sync-gather
