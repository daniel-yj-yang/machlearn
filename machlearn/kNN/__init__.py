# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from ._kNN import kNN_classifier, demo

# this is for "from <package_name>.kNN import *"
__all__ = ["kNN_classifier",
           "demo",]
