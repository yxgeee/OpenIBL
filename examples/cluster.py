from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys, os
import math
import h5py
from sklearn.cluster import KMeans

import torch
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ibl import datasets
from ibl import models
from ibl.evaluators import extract_features, pairwise_distance
from ibl.utils.data import get_transformer_train, get_transformer_test
from ibl.utils.data.sampler import SubsetRandomSampler
from ibl.utils.data.preprocessor import Preprocessor
from ibl.utils.logging import Logger
from ibl.utils.serialization import load_checkpoint, copy_state_dict


def get_data(args, nIm):
    root = osp.join(args.data_dir, args.dataset)
    dataset = datasets.create(args.dataset, root, scale='30k')
    cluster_set = list(set(dataset.q_train) | set(dataset.db_train))

    transformer = get_transformer_test(args.height, args.width)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    cluster_loader = DataLoader(Preprocessor(cluster_set, root=dataset.images_dir, transform=transformer),
                            batch_size=args.batch_size, num_workers=args.workers, sampler=sampler,
                            shuffle=False, pin_memory=True)

    return dataset, cluster_loader

def get_model(args):
    model = models.create(args.arch, pretrained=True, cut_at_pooling=True, matconvnet='logs/vd16_offtheshelf_conv5_3_max.pth')
    model.cuda()
    model = nn.DataParallel(model)
    return model

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)

def main_worker(args):
    cudnn.benchmark = True

    print("==========\nArgs:{}\n==========".format(args))

    nDescriptors = 50000
    nPerImage = 100
    nIm = math.ceil(nDescriptors/nPerImage)

    # Create data loaders
    dataset, data_loader = get_data(args, nIm)

    # Create model
    model = get_model(args)
    encoder_dim = model.module.feature_dim

	# Load from resume
    if args.resume:
        print('Loading weights from {}'.format(args.resume))
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)

    if not osp.exists(osp.join(args.logs_dir)):
        os.makedirs(osp.join(args.logs_dir))

    initcache = osp.join(args.logs_dir, args.arch + '_' + args.dataset + '_' + str(args.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5:
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors",
                        [nDescriptors, encoder_dim],
                        dtype=np.float32)

            for iteration, (input, _, _, _, _) in enumerate(data_loader, 1):
                input = input.cuda()
                image_descriptors = model(input)
                # normalization is IMPORTANT!
                image_descriptors = F.normalize(image_descriptors, p=2, dim=1).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration-1)*args.batch_size*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if (iteration % args.print_freq == 0) or (len(data_loader) <= args.print_freq):
                    print("==> Batch ({}/{})".format(iteration, math.ceil(nIm/args.batch_size)), flush=True)
                del input, image_descriptors

        print('====> Clustering')
        niter = 100
        kmeans = KMeans(n_clusters=args.num_clusters, max_iter=niter, random_state=args.seed).fit(dbFeat[...])

        print('====> Storing centroids', kmeans.cluster_centers_.shape)
        h5.create_dataset('centroids', data=kmeans.cluster_centers_)
        print('====> Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VLAD centers initialization clustering")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='pitts',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help="tuple numbers in a batch")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--num-clusters', type=int, default=64)
    parser.add_argument('--height', type=int, default=480, help="input height")
    parser.add_argument('--width', type=int, default=640, help="input width")
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--print-freq', type=int, default=10)
    # model
    parser.add_argument('-a', '--arch', type=str, default='vgg16',
                        choices=models.names())
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '..', 'logs'))
    main()
