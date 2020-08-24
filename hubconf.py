dependencies = ['torch']
import torch
from ibl import models

def vgg16_netvlad(pretrained=False):
    base_model = models.create('vgg16', pretrained=False)
    pool_layer = models.create('netvlad', dim=base_model.feature_dim)
    model = models.create('embednetpca', base_model, pool_layer)
    model.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/yxgeee/OpenIBL/releases/download/v0.1.0-beta/vgg16_netvlad.pth'))
    return model
