from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import h5py
import scipy.io
import copy

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed as datadist

from ibl import datasets
from ibl import models
from ibl.trainers import SFRSTrainer
from ibl.evaluators import Evaluator, extract_features, pairwise_distance
from ibl.utils.data import IterLoader, get_transformer_train, get_transformer_test
from ibl.utils.data.sampler import DistributedRandomDiffTupleSampler, DistributedSliceSampler
from ibl.utils.data.preprocessor import Preprocessor
from ibl.utils.logging import Logger
from ibl.pca import PCA
from ibl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from ibl.utils.dist_utils import init_dist, synchronize, convert_sync_bn
from ibl.utils.rerank import re_ranking


start_epoch = start_gen = best_recall5 = 0

def get_data(args, iters):
    """
    Get training dataset.

    Args:
        iters: (int): write your description
    """
    root = osp.join(args.data_dir, args.dataset)
    dataset = datasets.create(args.dataset, root, scale=args.scale)

    train_transformer = get_transformer_train(args.height, args.width)
    test_transformer = get_transformer_test(args.height, args.width)

    sampler = DistributedRandomDiffTupleSampler(dataset.q_train, dataset.db_train, dataset.train_pos, dataset.train_neg,
                                    pos_num=args.pos_num, pos_pool=args.pos_pool, neg_num=args.neg_num, neg_pool=args.neg_pool)

    train_loader = IterLoader(
                DataLoader(Preprocessor(dataset.q_train+dataset.db_train, root=dataset.images_dir,
                                        transform=train_transformer),
                            batch_size=args.tuple_size, num_workers=args.workers, sampler=sampler,
                            shuffle=False, pin_memory=True, drop_last=True), length=iters)

    train_extract_loader = DataLoader(
        Preprocessor(sorted(list(set(dataset.q_train) | set(dataset.db_train))),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=args.test_batch_size, num_workers=args.workers,
        sampler=DistributedSliceSampler(sorted(list(set(dataset.q_train) | set(dataset.db_train)))),
        shuffle=False, pin_memory=True)

    val_loader = DataLoader(
        Preprocessor(sorted(list(set(dataset.q_val) | set(dataset.db_val))),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=args.test_batch_size, num_workers=args.workers,
        sampler=DistributedSliceSampler(sorted(list(set(dataset.q_val) | set(dataset.db_val)))),
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(sorted(list(set(dataset.q_test) | set(dataset.db_test))),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=args.test_batch_size, num_workers=args.workers,
        sampler=DistributedSliceSampler(sorted(list(set(dataset.q_test) | set(dataset.db_test)))),
        shuffle=False, pin_memory=True)

    return dataset, train_loader, val_loader, test_loader, sampler, train_extract_loader

def update_sampler(sampler, model, loader, query, gallery, sub_set, rerank=False,
                        vlad=True, gpu=None, sync_gather=False, lambda_value=0.1):
    """
    Update sampler. sampler.

    Args:
        sampler: (todo): write your description
        model: (todo): write your description
        loader: (todo): write your description
        query: (str): write your description
        gallery: (todo): write your description
        sub_set: (todo): write your description
        rerank: (todo): write your description
        vlad: (todo): write your description
        gpu: (todo): write your description
        sync_gather: (str): write your description
        lambda_value: (todo): write your description
    """
    if (dist.get_rank()==0):
        print ("===> Start extracting features for sorting gallery")
    features = extract_features(model, loader, sorted(list(set(query) | set(gallery))),
                                vlad=vlad, gpu=gpu, sync_gather=sync_gather)
    distmat, _, _ = pairwise_distance(features, query, gallery)
    if rerank:
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat_jac = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy(),
                                                    k1=20, k2=1, lambda_value=lambda_value)
        distmat_jac = torch.from_numpy(distmat_jac)
        del distmat_qq, distmat_gg
    else:
        distmat_jac = distmat
    del features
    if (dist.get_rank()==0):
        print ("===> Start sorting gallery")
    sampler.sort_gallery(distmat, distmat_jac, sub_set)
    del distmat, distmat_jac

def get_model(args):
    """
    Get a model.

    Args:
    """
    base_model = models.create(args.arch, train_layers=args.layers, matconvnet='logs/vd16_offtheshelf_conv5_3_max.pth')
    pool_layer = models.create('netvlad', dim=base_model.feature_dim)
    initcache = osp.join(args.init_dir, args.arch + '_' + args.dataset + '_' + str(args.num_clusters) + '_desc_cen.hdf5')
    if (dist.get_rank()==0):
        print ('Loading centroids from {}'.format(initcache))
    with h5py.File(initcache, mode='r') as h5:
        pool_layer.clsts = h5.get("centroids")[...]
        pool_layer.traindescs = h5.get("descriptors")[...]
        pool_layer._init_params()

    model = models.create('embedregionnet', base_model, pool_layer, tuple_size=args.tuple_size)

    if (args.syncbn):
        convert_sync_bn(model)
    model.cuda(args.gpu)
    model = nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
            )
    return model

def main():
    """
    Main entry point.

    Args:
    """
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    """
    The main function.

    Args:
    """
    global start_epoch, start_gen, best_recall5
    init_dist(args.launcher, args)
    synchronize()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False

    print("Use GPU: {} for training, rank no.{} of world_size {}"
          .format(args.gpu, args.rank, args.world_size))

    if (args.rank==0):
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset, train_loader, val_loader, test_loader, sampler, train_extract_loader = get_data(args, iters)

    # Create model
    model = get_model(args)
    model_cache = get_model(args)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']+1
        start_gen = checkpoint['generation']
        best_recall5 = checkpoint['best_recall5']
        if (args.rank==0):
            print("=> Start epoch {}  best recall5 {:.1%}"
                  .format(start_epoch, best_recall5))

    # Evaluator
    evaluator = Evaluator(model)

    if (args.rank==0):
        print("Test the initial model:")
    recalls = evaluator.evaluate(val_loader, sorted(list(set(dataset.q_val) | set(dataset.db_val))),
                        dataset.q_val, dataset.db_val, dataset.val_pos,
                        vlad=True, gpu=args.gpu, sync_gather=args.sync_gather)

    # Trainer
    trainer = SFRSTrainer(model, model_cache, margin=args.margin**0.5,
                                    neg_num=args.neg_num, gpu=args.gpu, temp=args.temperature)
    if ((args.cache_size<args.tuple_size) or (args.cache_size>len(dataset.q_train))):
        args.cache_size = len(dataset.q_train)

    for gen in range(start_gen, args.generations):
        # Update model cache and init model
        model_cache.load_state_dict(model.state_dict())
        model.module._init_params()

        # Optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

        if (gen==0):
            start_epoch = args.epochs-1

        for epoch in range(start_epoch, args.epochs):

            sampler.set_epoch(args.seed+epoch)
            if (epoch%args.step_size==0):
                args.cache_size = args.cache_size * (2 ** (epoch // args.step_size))

            g = torch.Generator()
            g.manual_seed(args.seed+epoch)
            subset_indices = torch.randperm(len(dataset.q_train), generator=g).long().split(args.cache_size)

            for subid, subset in enumerate(subset_indices):
                update_sampler(sampler, model, train_extract_loader, dataset.q_train, dataset.db_train, subset.tolist(),
                                rerank=(gen>0), vlad=True, gpu=args.gpu, sync_gather=args.sync_gather)
                synchronize()

                trainer.train(gen, epoch, subid, train_loader, optimizer,
                                train_iters=len(train_loader), print_freq=args.print_freq,
                                lambda_soft=(args.soft_weight if gen>0 else 0), loss_type=args.loss_type)
                synchronize()

            if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
                recalls = evaluator.evaluate(val_loader, sorted(list(set(dataset.q_val) | set(dataset.db_val))),
                                        dataset.q_val, dataset.db_val, dataset.val_pos,
                                        vlad=True, gpu=args.gpu, sync_gather=args.sync_gather)

                is_best = recalls[1] > best_recall5
                best_recall5 = max(recalls[1], best_recall5)

                if (args.rank==0):
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'generation': gen,
                        'best_recall5': best_recall5,
                    }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint'+str(gen)+'_'+str(epoch)+'.pth.tar'))
                    print('\n * Finished generation {:3d} epoch {:3d} recall@1: {:5.1%}  recall@5: {:5.1%}  recall@10: {:5.1%}  best@5: {:5.1%}{}\n'.
                          format(gen, epoch, recalls[0], recalls[1], recalls[2], best_recall5, ' *' if is_best else ''))

            lr_scheduler.step()
            synchronize()

        start_epoch = 0

    # final inference
    if (args.rank==0):
        print("Performing PCA reduction on the best model:")
    model.load_state_dict(load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))['state_dict'])
    pca_parameters_path = osp.join(args.logs_dir, 'pca_params_model_best.h5')
    pca = PCA(args.features, (not args.nowhiten), pca_parameters_path)
    dict_f = extract_features(model, train_extract_loader, sorted(list(set(dataset.q_train) | set(dataset.db_train))),
                                vlad=True, gpu=args.gpu, sync_gather=args.sync_gather)
    features = list(dict_f.values())
    if (len(features)>10000):
        features = random.sample(features, 10000)
    features = torch.stack(features)
    if (args.rank==0):
        pca.train(features)
    synchronize()
    del features
    if (args.rank==0):
        print("Testing on Pitts30k-test:")
    evaluator.evaluate(test_loader, sorted(list(set(dataset.q_test) | set(dataset.db_test))),
                dataset.q_test, dataset.db_test, dataset.test_pos,
                vlad=True, pca=pca, gpu=args.gpu, sync_gather=args.sync_gather)
    synchronize()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SFRS training")
    parser.add_argument('--launcher', type=str,
                        choices=['none', 'pytorch', 'slurm'],
                        default='none', help='job launcher')
    parser.add_argument('--tcp-port', type=str, default='5017')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='pitts',
                        choices=datasets.names())
    parser.add_argument('--scale', type=str, default='30k')
    parser.add_argument('--tuple-size', type=int, default=1,
                        help="tuple numbers in a batch")
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help="tuple numbers in a batch")
    parser.add_argument('--cache-size', type=int, default=1000)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=480, help="input height")
    parser.add_argument('--width', type=int, default=640, help="input width")
    parser.add_argument('--num-clusters', type=int, default=64)
    parser.add_argument('--pos-num', type=int, default=10)
    parser.add_argument('--pos-pool', type=int, default=20)
    parser.add_argument('--neg-num', type=int, default=10,
                        help="negative instances for one anchor in a tuple")
    parser.add_argument('--neg-pool', type=int, default=1000)
    # model
    parser.add_argument('-a', '--arch', type=str, default='vgg16',
                        choices=models.names())
    parser.add_argument('--layers', type=str, default='conv5')
    parser.add_argument('--nowhiten', action='store_true')
    parser.add_argument('--syncbn', action='store_true')
    parser.add_argument('--sync-gather', action='store_true')
    parser.add_argument('--features', type=int, default=4096)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--step-size', type=int, default=5)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--generations', type=int, default=4)
    parser.add_argument('--loss-type', type=str, default='sare_ind')
    parser.add_argument('--temperature', nargs='+', type=float, default=[0.07,0.07,0.06,0.05])
    parser.add_argument('--soft-weight', type=float, default=0.5)
    parser.add_argument('--iters', type=int, default=0)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--margin', type=float, default=0.1, help='margin for the triplet loss with batch hard')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--init-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '..', 'logs'))
    main()
