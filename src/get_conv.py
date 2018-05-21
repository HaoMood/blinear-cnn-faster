#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Get relu5-3 features for CUB/Aircraft/Cars dataset.

Used for the fc process to speed up training.
"""


import os

import torch
import torchvision

import cub200

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True


__all__ = ['VGGManager']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-03-04'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-05-19'
__version__ = '13.1'


class VGGManager(object):
    """Manager class to extract VGG-16 relu5-3 features.

    Attributes:
        _paths, dict<str, str>: Useful paths.
        _net, torch.nn.Module: VGG-16 truncated at relu5-3.
        _train_loader, torch.utils.data.DataLoader: Training data.
        _test_loader, torch.utils.data.DataLoader: Testing data.
    """
    def __init__(self, paths):
        """Prepare the network and data.

        Args:
            paths, dict<str, str>: Useful paths.
        """
        print('Prepare the network and data.')

        # Configurations.
        self._paths = paths

        # Network.
        self._net = torchvision.models.vgg16(pretrained=True).features
        self._net = torch.nn.Sequential(*list(self._net.children())[:-2])
        self._net = self._net.cuda()

        # Data.
        # NOTE: Resize such that the short edge is 448, and then ceter crop 448.
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(448, 448)),
            # torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(448, 448)),
            # torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        train_data = cub200.CUB200(
            root=self._paths['cub200'], train=True, transform=train_transforms,
            download=True)
        test_data = cub200.CUB200(
            root=self._paths['cub200'], train=False, transform=test_transforms,
            download=True)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=False)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=False)

    def getFeature(self, phase, size):
        """Get relu5-3 features and save it onto disk.

        Args:
            phase, str: Train or test.
            size, int: Dataset size.
        """
        print('Get relu5-3 feaures for %s data.' % phase)
        if phase not in ['train', 'test']:
            raise RuntimeError('phase should be train/test.')
        with torch.no_grad():
            all_data = []  # list<torch.Tensor>
            all_label = []  # list<int>
            data_loader = (self._train_loader if phase == 'train'
                           else self._test_loader)
            for instance, label in data_loader:
                # Data.
                instance = instance.cuda()
                assert instance.size() == (1, 3, 448, 448)
                assert label.size() == (1,)

                # Forward pass
                feature = self._net(instance)
                assert feature.size() == (1, 512, 28, 28)

                all_data.append(torch.squeeze(feature, dim=0).cpu())
                all_label.append(label.item())
            assert len(all_data) == size and len(all_label) == size
            torch.save((all_data, all_label), os.path.join(
                self._paths['cub200'], 'relu5-3', '%s.pth' % phase))

def main():
    """The main function."""
    project_root = os.popen('pwd').read().strip()
    paths = {
        'cub200': os.path.join(project_root, 'data', 'cub200'),
        'aircraft': os.path.join(project_root, 'data', 'aircraft'),
        'cars': os.path.join(project_root, 'data', 'cars'),
    }
    for d in paths:
        assert os.path.isdir(paths[d])

    manager = VGGManager(paths)
    manager.getFeature('train', 5994)
    manager.getFeature('test', 5794)


if __name__ == '__main__':
    main()
