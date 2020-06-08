from __future__ import absolute_import
import os
import re
import os.path as osp
import numpy as np
import random
import math
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader, Dataset


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, x, y = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if (self.transform is not None):
            img = self.transform(img)

        return img, fname, pid, x, y
