# -*- coding: utf-8 -*
"""This module is served as torchvision.datasets to load Aircraft dataset.

This file is modified from:
    https://github.com/vishwakftw/vision.
"""


import os
import pickle

import PIL.Image
import torch


__all__ = ['Aircraft', 'AircraftReLU']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-04-19'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-04-19'
__version__ = '11.1'


class Aircraft(torch.utils.data.Dataset):
    """Aircraft dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _target_transform, callable: A function/transform that takes in the
            target and transforms it.
        _train_data, list of np.ndarray.
        _train_labels, list of int.
        _test_data, list of np.ndarray.
        _test_labels, list of int.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._train = train
        self._transform = transform
        self._target_transform = target_transform

        if self._checkIntegrity():
            print('Files already downloaded and verified.')
        elif download:
            url = None
            self._download(url)
            self._extract()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        # Now load the picked data.
        if self._train:
            self._train_data, self._train_labels = pickle.load(open(
                os.path.join(self._root, 'processed/train.pkl'), 'rb'))
            assert (len(self._train_data) == 6667
                    and len(self._train_labels) == 6667)
        else:
            self._test_data, self._test_labels = pickle.load(open(
                os.path.join(self._root, 'processed/test.pkl'), 'rb'))
            assert (len(self._test_data) == 3333
                    and len(self._test_labels) == 3333)

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
        if self._train:
            image, target = self._train_data[index], self._train_labels[index]
        else:
            image, target = self._test_data[index], self._test_labels[index]
        # Doing this so that it is consistent with all other datasets.
        image = PIL.Image.fromarray(image)

        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            target = self._target_transform(target)

        return image, target

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _checkIntegrity(self):
        """Check whether we have already processed the data.

        Returns:
            flag, bool: True if we have already processed the data.
        """
        return (
            os.path.isfile(os.path.join(self._root, 'processed/train.pkl'))
            and os.path.isfile(os.path.join(self._root, 'processed/test.pkl')))

    def _download(self, url):
        raise NotImplementedError

    def _extract(self):
        raise NotImplementedError


class AircraftReLU(torch.utils.data.Dataset):
    """Aircraft relu5-3 dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _train_data, list<torch.Tensor>.
        _train_labels, list<int>.
        _test_data, list<torch.Tensor>.
        _test_labels, list<int>.
    """
    def __init__(self, root, train=True):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._train = train

        if self._checkIntegrity():
            print('Aircraft relu5-3 features already prepared.')
        else:
            raise RuntimeError('Aircraft relu5-3 Dataset not found.'
                'You need to prepare it in advance.')

        # Now load the picked data.
        if self._train:
            self._train_data, self._train_labels = pickle.load(open(
                os.path.join(self._root, 'relu5-3/train.pkl'), 'rb'))
            assert (len(self._train_data) == 6667
                    and len(self._train_labels) == 6667)
        else:
            self._test_data, self._test_labels = pickle.load(open(
                os.path.join(self._root, 'relu5-3/test.pkl'), 'rb'))
            assert (len(self._test_data) == 3333
                    and len(self._test_labels) == 3333)

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            feature, torch.Tensor: relu5-3 feature of the given index.
            target, int: target of the given index.
        """
        if self._train:
            return self._train_data[index], self._train_labels[index]
        return self._test_data[index], self._test_labels[index]

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _checkIntegrity(self):
        """Check whether we have already processed the data.

        Returns:
            flag, bool: True if we have already processed the data.
        """
        return (
            os.path.isfile(os.path.join(self._root, 'relu5-3/train.pkl'))
            and os.path.isfile(os.path.join(self._root, 'relu5-3/test.pkl')))
