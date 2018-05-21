# -*- coding: utf-8 -*-
"""Mean field B-CNN model."""


import torch
import torchvision

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benckmark = True


__all__ = ['BCNN']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-01-09'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-05-21'
__version__ = '13.7'


class BCNN(torch.nn.Module):
    """Mean field B-CNN model.

    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> mean field bilinear pooling
    -> fc.

    The network accepts a 3*448*448 input, and the relu5-3 activation has shape
    512*28*28 since we down-sample 4 times.

    Attributes:
        _is_all, bool: In the all/fc phase.
        features, torch.nn.Module: Convolution and pooling layers.
        bn, torch.nn.Module.
        gap_pool, torch.nn.Module.
        mf_relu, torch.nn.Module.
        mf_pool, torch.nn.Module.
        fc, torch.nn.Module.
    """
    def __init__(self, num_classes, is_all):
        """Declare all needed layers.

        Args:
            num_classes, int.
            is_all, bool: In the all/fc phase.
        """
        torch.nn.Module.__init__(self)
        self._is_all = is_all

        if self._is_all:
            # Convolution and pooling layers of VGG-16.
            self.features = torchvision.models.vgg16(pretrained=True).features
            self.features = torch.nn.Sequential(*list(self.features.children())
                                                [:-2])  # Remove pool5.

        # Mean filed pooling layer.
        self.relu5_3 = torch.nn.ReLU(inplace=False)

        # Classification layer.
        self.fc = torch.nn.Linear(
            in_features=512 * 512, out_features=num_classes, bias=True)

        if not self._is_all:
            self.apply(BCNN._initParameter)

    def _initParameter(module):
        """Initialize the weight and bias for each module.

        Args:
            module, torch.nn.Module.
        """
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, val=1.0)
            torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out',
                                          nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.Tensor (N*3*448*448).

        Returns:
            score, torch.Tensor (N*200).
        """
        # Input.
        N = X.size()[0]
        if self._is_all:
            assert X.size() == (N, 3, 448, 448)
            X = self.features(X)
        assert X.size() == (N, 512, 28, 28)

        # The main branch.
        X = self.relu5_3(X)
        assert X.size() == (N, 512, 28, 28)

        # Classical bilinear pooling.
        X = torch.reshape(X, (N, 512, 28 * 28))
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28 * 28)
        assert X.size() == (N, 512, 512)
        X = torch.reshape(X, (N, 512 * 512))

        # Normalization.
        # X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)

        # Classification.
        X = self.fc(X)
        return X
