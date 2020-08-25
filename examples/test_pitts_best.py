import argparse
import os.path as osp

import torch
from torch import nn
from torch.utils.data import DataLoader

from ibl import datasets
from ibl.evaluators import Evaluator
from ibl.utils.data import get_transformer_test
from ibl.utils.data.sampler import DistributedSliceSampler
from ibl.utils.data.preprocessor import Preprocessor
from ibl.utils.dist_utils import init_dist, synchronize


def get_data(args):
    root = osp.join(args.data_dir, 'pitts')
    dataset = datasets.create('pitts', root, scale=args.scale)

    test_transformer = get_transformer_test(args.height, args.width)

    test_loader_q = DataLoader(
        Preprocessor(dataset.q_test, root=dataset.images_dir, transform=test_transformer),
        batch_size=args.test_batch_size, num_workers=args.workers,
        sampler=DistributedSliceSampler(dataset.q_test),
        shuffle=False, pin_memory=True)

    test_loader_db = DataLoader(
        Preprocessor(dataset.db_test, root=dataset.images_dir, transform=test_transformer),
        batch_size=args.test_batch_size, num_workers=args.workers,
        sampler=DistributedSliceSampler(dataset.db_test),
        shuffle=False, pin_memory=True)

    return dataset, test_loader_q, test_loader_db

def vgg16_netvlad(pretrained=False):
    base_model = models.create('vgg16', pretrained=False)
    pool_layer = models.create('netvlad', dim=base_model.feature_dim)
    model = models.create('embednetpca', base_model, pool_layer)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/yxgeee/OpenIBL/releases/download/v0.1.0-beta/vgg16_netvlad.pth', map_location=torch.device('cpu')))
    return model

def get_model(args):
    model = vgg16_netvlad(pretrained=True)
    model.cuda(args.gpu)
    model = nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
            )
    return model

def main():
    args = parser.parse_args()
    init_dist(args.launcher, args)
    synchronize()
    print("Use GPU: {} for testing, rank no.{} of world_size {}"
          .format(args.gpu, args.rank, args.world_size))

    if (args.rank==0):
        print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset, test_loader_q, test_loader_db = get_data(args)

    # Create model
    model = get_model(args)

    # Evaluator
    evaluator = Evaluator(model)

    if (args.rank==0):
        print("Evaluate on the test set:")
    evaluator.evaluate(test_loader_q, sorted(list(set(dataset.q_test) | set(dataset.db_test))),
                        dataset.q_test, dataset.db_test, dataset.test_pos, gallery_loader=test_loader_db,
                        gpu=args.gpu, sync_gather=args.sync_gather, nms=False)
    synchronize()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-based localization testing")
    parser.add_argument('--launcher', type=str,
                        choices=['none', 'pytorch', 'slurm'],
                        default='none', help='job launcher')
    parser.add_argument('--tcp-port', type=str, default='5017')
    # data
    parser.add_argument('--scale', type=str, default='250k')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        help="tuple numbers in a batch")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=480, help="input height")
    parser.add_argument('--width', type=int, default=640, help="input width")
    # model
    parser.add_argument('--sync-gather', action='store_true')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir))
    main()
