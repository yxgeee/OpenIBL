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
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
            dataset: (todo): write your description
            root: (str): write your description
            transform: (str): write your description
        """
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        """
        Returns the number of the dataset.

        Args:
            self: (todo): write your description
        """
        return len(self.dataset)

    def __getitem__(self, indices):
        """
        Returns the item from index.

        Args:
            self: (todo): write your description
            indices: (int): write your description
        """
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        """
        Get single single singleton item.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        fname, pid, x, y = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if (self.transform is not None):
            img = self.transform(img)

        return img, fname, pid, x, y
