from __future__ import absolute_import

import torchvision.transforms as T

from .dataset import Dataset
from .preprocessor import Preprocessor

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

def get_transformer_train(height, width):
    train_transformer = [T.ColorJitter(0.7, 0.7, 0.7, 0.5),
                         T.Resize((height, width)),
                         T.ToTensor(),
                         T.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                    std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])]
    return T.Compose(train_transformer)

def get_transformer_test(height, width, tokyo=False):
    test_transformer = [T.Resize(max(height,width) if tokyo else (height, width)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                   std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])]
    return T.Compose(test_transformer)
