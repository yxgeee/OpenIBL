'''
calculate PCA for given features, and save output into file.
'''

import os
import sys
import time
import glob
import numpy as np
import h5py

import sklearn
import scipy.linalg as la
import scipy.sparse.linalg as sla

import torch
import torch.nn.functional as F
import torch.distributed as dist


class PCA():
    def __init__(self, pca_n_components=4096, pca_whitening=True,
                 pca_parameters_path='./logs/pca_params.h5'):
        self.pca_n_components = pca_n_components
        self.pca_whitening = pca_whitening
        self.pca_parameters_path = pca_parameters_path

    def train(self, x):
        '''training pca.
        Args:
            x: [N, dim] FloatTensor containing data which undergoes PCA/Whitening.
        '''

        print('calculating PCA parameters...')

        ### Referring to https://github.com/Relja/relja_matlab/blob/master/relja_PCA.m

        x = x.t()
        nPoints = x.size(1)
        nDims = x.size(0)

        # x = x.double()
        mu = x.mean(1).unsqueeze(1)
        x = x - mu

        if (nDims<=nPoints):
            doDual = False
            x2 = torch.matmul(x, x.t()) / (nPoints - 1)
        else:
            doDual = True
            x2 = torch.matmul(x.t(), x) / (nPoints - 1)

        L, U = torch.symeig(x2, eigenvectors=True)
        if (self.pca_n_components < x2.size(0)):
            k_indices = torch.argsort(L, descending=True)[:self.pca_n_components]
            L = torch.index_select(L, 0, k_indices)
            U = torch.index_select(U, 1, k_indices)

        lams = L
        lams[lams<1e-9] = 1e-9

        if (doDual):
            U = torch.matmul(x, torch.matmul(U, torch.diag(1./torch.sqrt(lams))/np.sqrt(nPoints-1)))

        Utmu= torch.matmul(U.t(), mu)

        U, lams, mu, Utmu = U.numpy(), lams.numpy(), mu.numpy(), Utmu.numpy()

        # save as h5 file.
        print('================= PCA RESULT ==================')
        print('U: {}'.format(U.shape))
        print('lams: {}'.format(lams.shape))
        print('mu: {}'.format(mu.shape))
        print('Utmu: {}'.format(Utmu.shape))
        print('===============================================')

        # save features, labels to h5 file.
        filename = os.path.join(self.pca_parameters_path)
        h5file = h5py.File(filename, 'w')
        h5file.create_dataset('U', data=U)
        h5file.create_dataset('lams', data=lams)
        h5file.create_dataset('mu', data=mu)
        h5file.create_dataset('Utmu', data=Utmu)
        h5file.close()

    def load(self, gpu=None):
        try:
            rank = dist.get_rank()
        except:
            rank = 0
        if (rank==0):
            print('load PCA parameters...')
        h5file = h5py.File(self.pca_parameters_path, 'r')

        ### Referring to https://github.com/Relja/relja_matlab/blob/master/relja_PCA.m
        U = h5file['.']['U'][...][:, :self.pca_n_components]
        lams = h5file['.']['lams'][...][:self.pca_n_components]
        mu = h5file['.']['mu'][...]
        Utmu = h5file['.']['Utmu'][...]

        if (self.pca_whitening):
            U = np.matmul(U, np.diag(1./np.sqrt(lams)))
        Utmu = np.matmul(U.T, mu)

        self.weight = torch.from_numpy(U.T).view(self.pca_n_components, -1, 1, 1).float().cuda(gpu)
        self.bias = torch.from_numpy(-Utmu).view(-1).float().cuda(gpu)

    def infer(self, data):
        '''apply PCA/Whitening to data.
        Args:
            data: [N, dim] FloatTensor containing data which undergoes PCA/Whitening.
        Returns:
            output: [N, output_dim] FloatTensor with output of PCA/Whitening operation.
        '''

        ### Referring to https://github.com/Relja/netvlad/blob/master/addPCA.m
        N, D = data.size()
        data = data.view(N, D, 1, 1)
        output = F.conv2d(data, self.weight, bias=self.bias, stride=1, padding=0).view(N, -1)

        output = F.normalize(output, p=2, dim=-1) # IMPORTANT!
        assert (output.size(1)==self.pca_n_components)
        return output
