#!/usr/bin/env python
"""Get relu5-3 features for CUB/Aircraft dataset.

Used for the fc process to speed up training.
"""


import os
import pickle

import torch
import torchvision

import cub200


__all__ = ['VGGManager']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-03-04'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-04-19'
__version__ = '11.1'


class VGGManager(object):
    """Manager class to extract VGG-16 relu5-3 features.

    Attributes:
        _path, dict<str, str>: Useful paths.
        _net, torch.nn.Module: VGG-16 truncated at relu5-3.
        _train_loader, torch.utils.data.DataLoader: Training data.
        _test_loader, torch.utils.data.DataLoader: Testing data.
    """
    def __init__(self, path):
        """Prepare the network and data.

        Args:
            path, dict<str, str>: Useful paths.
        """
        print('Prepare the network and data.')
        self._path = path
        self._net = torchvision.models.vgg16(pretrained=True).features
        # Remove pool5.
        self._net = torch.nn.Sequential(*list(self._net.children())[:-1])
        self._net = self._net.cuda()

        # NOTE: Resize such that the short edge is 448, and then ceter crop 448.
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        train_data = cub200.CUB200(
            root=self._path['cub200'], train=True, download=True,
            transform=train_transforms)
        test_data = cub200.CUB200(
            root=self._path['cub200'], train=False, download=True,
            transform=test_transforms)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=False)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=False)

    def getFeature(self):
        """Get relu5-3 features and save it onto disk."""
        print('Get relu5-3 feaures for training data.')
        train_data = []  # list<torch.Tensor>
        train_label = []  # list<int>
        for X, y in self._train_loader:
            X = torch.autograd.Variable(X.cuda())
            # y = torch.autograd.Variable(y)
            assert X.size() == (1, 3, 448, 448)
            feature = self._net(X)
            assert feature.size() == (1, 512, 28, 28)
            assert y.size() == (1,)
            train_data.append(torch.squeeze(feature, dim=0).cpu().data)
            train_label.append(y[0])
        assert len(train_data) == 6667 and len(train_label) == 6667
        pickle.dump(
            (train_data, train_label),
            open(os.path.join(self._path['cub200'],
                              'relu5-3/train.pkl'), 'wb'))

        print('Get relu5-3 feaures for test data.')
        test_data = []  # list<torch.Tensor>
        test_label = []  # list<int>
        for X, y in self._test_loader:
            X = torch.autograd.Variable(X.cuda())
            # y = torch.autograd.Variable(y)
            assert X.size() == (1, 3, 448, 448)
            feature = self._net(X)
            assert feature.size() == (1, 512, 28, 28)
            assert y.size() == (1,)
            test_data.append(torch.squeeze(feature, dim=0).cpu().data)
            test_label.append(y[0])
        assert len(test_data) == 3333 and len(test_label) == 3333
        pickle.dump(
            (test_data, test_label),
            open(os.path.join(self._path['cub200'],
                              'relu5-3/test.pkl'), 'wb'))


def main():
    """The main function."""
    project_root = os.popen('pwd').read().strip()
    path = {
        'cub200': os.path.join(project_root, 'data/cub200'),
        'aircraft': os.path.join(project_root, 'data/aircraft'),
    }
    for d in path:
        assert os.path.isdir(path[d])

    manager = VGGManager(path)
    manager.getFeature()


if __name__ == '__main__':
    main()
