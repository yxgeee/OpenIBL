## Train

All the training details (hyper-parameters, trained layers, backbones, etc.) strictly follow the original MatConvNet version of NetVLAD and SARE. **Note:** the results of all three methods (SFRS, NetVLAD, SARE) can be reproduced by training on Pitts30k-train and directly testing on the other datasets.

The default scripts adopt 4 GPUs (require ~11G per GPU) for training, where each GPU loads one tuple (anchor, positive(s), negatives).
+ In case you want to fasten training, enlarge `GPUS` for more GPUs, or enlarge the `--tuple-size` for more tuples on one GPU;
+ In case your GPU does not have enough memory (e.g. <11G), reduce `--pos-num` (only for SFRS) or `--neg-num` for fewer positives or negatives in one tuple.

#### PyTorch launcher: single-node multi-gpu distributed training

NetVLAD:
```shell
./scripts/train_baseline_dist.sh triplet
```

SARE:
```shell
./scripts/train_baseline_dist.sh sare_ind
# or
./scripts/train_baseline_dist.sh sare_joint
```

SFRS (state-of-the-art):
```shell
./scripts/train_sfrs_dist.sh
```

#### Slurm launcher: single/multi-node multi-gpu distributed training

Change `GPUS` and `GPUS_PER_NODE` accordingly in the scripts for your need.

NetVLAD:
```shell
./scripts/train_baseline_slurm.sh <PARTITION NAME> triplet
```

SARE:
```shell
./scripts/train_baseline_slurm.sh <PARTITION NAME> sare_ind
# or
./scripts/train_baseline_slurm.sh <PARTITION NAME> sare_joint
```

SFRS (state-of-the-art):
```shell
./scripts/train_sfrs_slurm.sh <PARTITION NAME>
```

## Test

Trained models can be found in [MODEL_ZOO.md](MODEL_ZOO.md).

During testing, the python scripts will automatically compute the PCA weights from Pitts30k-train or directly load from local files. Generally, `model_best.pth.tar` which is selected by validation in the training performs the best.

The default scripts adopt 8 GPUs (require ~11G per GPU) for testing.
+ In case you want to fasten testing, enlarge `GPUS` for more GPUs, or enlarge the `--test-batch-size` for larger batch size on one GPU, or add `--sync-gather` for faster gathering from multiple threads;
+ In case your GPU does not have enough memory (e.g. <11G), reduce `--test-batch-size` for smaller batch size on one GPU.

#### PyTorch launcher: single-node multi-gpu distributed testing

Pitts250k-test:
```shell
./scripts/test_dist.sh <PATH TO MODEL> pitts 250k
```

Pitts30k-test:
```shell
./scripts/test_dist.sh <PATH TO MODEL> pitts 30k
```

Tokyo 24/7:
```shell
./scripts/test_dist.sh <PATH TO MODEL> tokyo
```

#### Slurm launcher: single/multi-node multi-gpu distributed testing

Pitts250k-test:
```shell
./scripts/test_slurm.sh <PARTITION NAME> <PATH TO MODEL> pitts 250k
```

Pitts30k-test:
```shell
./scripts/test_slurm.sh <PARTITION NAME> <PATH TO MODEL> pitts 30k
```

Tokyo 24/7:
```shell
./scripts/test_slurm.sh <PARTITION NAME> <PATH TO MODEL> tokyo
```
