from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)
from torch.utils.data import BatchSampler
import torch.distributed as dist

class DistributedRandomTupleSampler(Sampler):
    def __init__(self, query_source, gallery_source, pos_list, neg_list,
                neg_num=10, neg_pool=1000, sub_length=None, num_replicas=None, rank=None):
        if(num_replicas is None):
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if(rank is None):
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.query_source = query_source
        self.gallery_source = gallery_source
        self.pos_list = pos_list
        self.neg_list = neg_list
        self.neg_num = neg_num
        self.neg_pool = neg_pool
        self.sub_set = torch.arange(len(query_source)).tolist()
        self.sub_length = sub_length

        if (self.sub_length is None):
            self.sub_length = len(query_source)
            self.sub_length_dist = int(math.ceil(self.sub_length * 1.0 / self.num_replicas))
            self.total_size = self.sub_length_dist * self.num_replicas

        self.sort_idx, self.neg_cache = None, [[]]*len(self.query_source)

    def sort_gallery(self, distmat, sub_set):
        assert(distmat.shape[0]==len(self.query_source))
        assert(distmat.shape[1]==len(self.gallery_source))
        self.sort_idx = torch.argsort(distmat, dim=1)
        self.sub_set = sub_set
        self.sub_length = len(self.sub_set)

        self.sub_length_dist = int(math.ceil(self.sub_length * 1.0 / self.num_replicas))
        self.total_size = self.sub_length_dist * self.num_replicas

    def __len__(self):
        return self.sub_length_dist

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        indices = torch.arange(self.sub_length).long().tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.sub_length_dist

        for i in indices:
            anchor_index = self.sub_set[i]
            # easiest positive
            pos_indices = [x for x in self.sort_idx[anchor_index].tolist() if x in self.pos_list[anchor_index]]
            pos_index = pos_indices[0]
            # hardest negatives from a random image pool and previous epoch
            neg_candidates = [x for x in self.sort_idx[anchor_index].tolist() if x not in self.neg_list[anchor_index]]
            neg_pool_index = random.sample(range(len(neg_candidates)), min(self.neg_pool,len(neg_candidates)))
            neg_cache_index = [neg_candidates.index(i) for i in self.neg_cache[anchor_index]]
            neg_pool_index = sorted(list(set(neg_pool_index) | set(neg_cache_index)))
            neg_indices = [neg_candidates[i] for i in neg_pool_index[:self.neg_num]]
            self.neg_cache[anchor_index] = neg_indices
            assert(len(neg_indices)==self.neg_num)

            iters = [anchor_index, pos_index + len(self.query_source)]
            iters += [n + len(self.query_source) for n in neg_indices]
            yield iters


class DistributedRandomDiffTupleSampler(Sampler):
    def __init__(self, query_source, gallery_source, pos_list, neg_list,
                pos_num=10, pos_pool=20, neg_num=10, neg_pool=1000,
                sub_length=None, num_replicas=None, rank=None):
        if(num_replicas is None):
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if(rank is None):
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.query_source = query_source
        self.gallery_source = gallery_source
        self.pos_list = pos_list
        self.neg_list = neg_list
        self.pos_num = pos_num
        self.pos_pool = pos_pool
        self.neg_num = neg_num
        self.neg_pool = neg_pool
        self.sub_set = torch.arange(len(query_source)).tolist()
        self.sub_length = sub_length

        if (self.sub_length is None):
            self.sub_length = len(query_source)
            self.sub_length_dist = int(math.ceil(self.sub_length * 1.0 / self.num_replicas))
            self.total_size = self.sub_length_dist * self.num_replicas

        self.sort_idx, self.distmat_jac, self.neg_cache = None, None, [[]]*len(self.query_source)

    def sort_gallery(self, distmat, distmat_jac, sub_set):
        assert(distmat.shape[0]==len(self.query_source))
        assert(distmat.shape[1]==len(self.gallery_source))
        self.sort_idx = torch.argsort(distmat, dim=1)
        self.distmat_jac = distmat_jac
        self.sub_set = sub_set
        self.sub_length = len(self.sub_set)

        self.sub_length_dist = int(math.ceil(self.sub_length * 1.0 / self.num_replicas))
        self.total_size = self.sub_length_dist * self.num_replicas

    def __len__(self):
        return self.sub_length_dist

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        indices = torch.arange(self.sub_length).long().tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.sub_length_dist

        for i in indices:
            anchor_index = self.sub_set[i]
            # nearest positive

            ## use original distance
            pos_indices = [x for x in self.sort_idx[anchor_index].tolist() if x in self.pos_list[anchor_index]]
            pos_top1 = pos_indices[0]

            ## use jaccard distance
            pos_indices = pos_indices[:self.pos_pool]
            pos_jac_dist = self.distmat_jac[anchor_index][torch.Tensor(pos_indices).long()]

            pos_jac_inds = torch.argsort(pos_jac_dist, dim=0)
            inds_gap = torch.arange(pos_jac_inds.size(0))-pos_jac_inds

            inds_gap_neg = torch.arange(pos_jac_inds.size(0))[inds_gap<0]
            sort_gap_ind_neg = torch.argsort(inds_gap[inds_gap<0], dim=0)
            sort_gap_ind_neg = inds_gap_neg[sort_gap_ind_neg]

            sort_gap_ind_zero = torch.arange(pos_jac_inds.size(0))[inds_gap==0]
            sort_gap_ind = torch.cat((sort_gap_ind_neg,sort_gap_ind_zero), dim=0)[:self.pos_num]

            sort_gap_ind = pos_jac_inds[sort_gap_ind]
            sort_gap_ind = torch.Tensor(pos_indices).long()[sort_gap_ind]
            pos_indices = sort_gap_ind.tolist()

            # hardest negatives from a random image pool and previous epoch
            neg_candidates = [x for x in self.sort_idx[anchor_index].tolist() if x not in self.neg_list[anchor_index]]
            neg_pool_index = random.sample(range(len(neg_candidates)), min(self.neg_pool,len(neg_candidates)))
            neg_cache_index = [neg_candidates.index(i) for i in self.neg_cache[anchor_index]]
            neg_pool_index = sorted(list(set(neg_pool_index) | set(neg_cache_index)))
            neg_indices = [neg_candidates[i] for i in neg_pool_index[:self.neg_num]]
            self.neg_cache[anchor_index] = neg_indices
            assert(len(neg_indices)==self.neg_num)

            iters = [anchor_index, pos_top1 + len(self.query_source)]
            iters += [n + len(self.query_source) for n in neg_indices]
            iters += [p+len(self.query_source) for p in pos_indices]
            yield iters

class DistributedSliceSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        self.slices = torch.arange(len(self.dataset)).long().tolist()
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.slices += self.slices[:(self.total_size - len(self.slices))]
        assert len(self.slices) == self.total_size
        self.slices = torch.LongTensor(self.slices).split(self.num_samples)

    def __iter__(self):
        indices = self.slices[self.rank].tolist()
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
