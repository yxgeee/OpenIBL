import os
import subprocess
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

def init_dist(launcher, args, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        init_dist_pytorch(args, backend)
    elif launcher == 'slurm':
        init_dist_slurm(args, backend)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))

def init_dist_pytorch(args, backend="nccl"):
    args.rank = int(os.environ['LOCAL_RANK'])
    args.ngpus_per_node = torch.cuda.device_count()
    args.gpu = args.rank
    args.world_size = args.ngpus_per_node
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend=backend)

def init_dist_slurm(args, backend="nccl"):
    args.rank = int(os.environ['SLURM_PROCID'])
    args.world_size = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    args.ngpus_per_node = torch.cuda.device_count()
    args.gpu = args.rank % args.ngpus_per_node
    torch.cuda.set_device(args.gpu)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(args.tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)
    dist.init_process_group(backend=backend)
    args.total_gpus = dist.get_world_size()

def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(dist.new_group(rank_list[i]))
    group_size = world_size // num_groups
    print ("Rank no.{} start sync BN on the process group of {}".format(rank, rank_list[rank//group_size]))
    return groups[rank//group_size]

def convert_sync_bn(model, process_group=None, gpu=None):
    for _, (child_name, child) in enumerate(model.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            m = nn.SyncBatchNorm.convert_sync_batchnorm(child, process_group)
            if (gpu is not None):
                m = m.cuda(gpu)
            setattr(model, child_name, m)
        else:
            convert_sync_bn(child, process_group, gpu)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
