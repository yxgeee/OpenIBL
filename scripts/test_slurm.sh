#!/bin/sh
PARTITION=$1
GPUS=8
GPUS_PER_NODE=8

RESUME=$2
ARCH=vgg16

DATASET=${3-pitts}
SCALE=${4-250k}

if [ $# -lt 2 ]
  then
    echo "Arguments error: <PARTITION NAME> <MODEL PATH>"
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

srun --mpi=pmi2 -p ${PARTITION} -n${GPUS} \
     --gres=gpu:${GPUS_PER_NODE} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --job-name=test \
python -u examples/test.py --launcher slurm --tcp-port ${PORT} \
      -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
      --test-batch-size 24 -j 2 \
      --vlad --reduction \
      --resume ${RESUME}
      # --sync-gather
