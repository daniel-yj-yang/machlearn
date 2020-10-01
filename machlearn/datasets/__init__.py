# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from ._datasets import public_dataset
from ._dataset_methods import Fashion_MNIST_methods

# this is for "from <package_name>.datasets import *"
__all__ = ["public_dataset", "Fashion_MNIST_methods"]
