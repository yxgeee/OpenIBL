#!/bin/sh
PARTITION=$1
GPUS=4
GPUS_PER_NODE=4

DATASET=pitts
SCALE=30k
ARCH=vgg16
LAYERS=conv5
LOSS=sare_ind
LR=0.001

if [ $# -ne 1 ]
  then
    echo "Arguments error: <PARTITION NAME>"
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
     --job-name=sfrs \
python -u examples/netvlad_img_sfrs.py --launcher slurm --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --syncbn \
  --width 640 --height 480 --tuple-size 1 -j 2 --test-batch-size 16 \
  --neg-num 10  --pos-pool 20 --neg-pool 1000 --pos-num 10 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} --soft-weight 0.5 \
  --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 --generations 4 --temperature 0.07 0.07 0.06 0.05 \
  --logs-dir logs/netVLAD/${DATASET}${SCALE}-${ARCH}/${LAYERS}-${LOSS}-lr${LR}-tuple${GPUS}-SFRS
  # --sync-gather
