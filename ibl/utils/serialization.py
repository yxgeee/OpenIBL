from __future__ import print_function, absolute_import
import json
import os.path as osp
import shutil
from scipy.io import loadmat

import torch
import torch.distributed as dist
from torch.nn import Parameter

from .osutils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_mat(path, key='dbStruct'):
    mat = loadmat(path)
    ws = mat[key].item()
    return ws

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        try:
            rank = dist.get_rank()
        except:
            rank = 0
        if (rank==0):
            print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            try:
                rank = dist.get_rank()
            except:
                rank = 0
            if (rank==0):
                print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    try:
        rank = dist.get_rank()
    except:
        rank = 0
    if ((len(missing) > 0) and (rank==0)):
        print("missing keys in state_dict:", missing)

    return model
