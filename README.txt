     Mean field approximation of Bilinear CNN for Fine-grained recognition


DESCRIPTIONS
    After getting the deep descriptors of an image, bilinear pooling computes
    the sum of the outer product of those deep descriptors. Bilinear pooling
    captures all pairwise descriptor interactions, i.e., interactions of
    different part, in a translational invariant manner.

    This project aims at accelerating training at the first step. We extract
    VGG-16 relu5-3 features from ImageNet pre-trained model in advance and save
    them onto disk. At the first step, we train the model directly from the
    extracted relu5-3 features. We avoid feed forwarding convolution layers
    multiple times.


PREREQUIREMENTS
    Python3.6 with Numpy supported
    PyTorch


LAYOUT
    ./data/                 # Datasets
    ./doc/                  # Automatically generated documents
    ./src/                  # Source code


USAGE
    Step 1. Fine-tune the fc layer only.
    # Get relu5-3 features from VGG-16 ImageNet pre-trained model.
    # It gives 75.47% accuracy on CUB.
    $ CUDA_VISIBLE_DEVICES=0 ./src/get_conv.py
    $ CUDA_VISIBLE_DEVICES=0,1,2,3 ./src/train.py --base_lr 1e0 \
          --batch_size 64 --epochs 80 --weight_decay 1e-5 \
          | tee "[fc-] base_lr_1e0-weight_decay_1e-5_.log"

    Step 2. Fine-tune all layers.
    # It gives 84.41% accuracy on CUB.
    $ CUDA_VISIBLE_DEVICES=0,1,2,3 ./src/train.py --base_lr 1e-2 \
          --batch_size 64 --epochs 80 --weight_decay 1e-5 \
          --pretrained "bcnn_fc_epoch_.pth" \
          | tee "[all-] base_lr_1e-2-weight_decay_1e-5.log"


AUTHOR
    Hao Zhang: zhangh0214@gmail.com


LICENSE
    CC BY-SA 3.0
