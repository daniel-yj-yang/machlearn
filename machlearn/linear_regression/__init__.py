# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from ._linear_regression import demo, LinearReg_normal_equation, LinearReg_statsmodels, demo_regularization

# this is for "from <package_name>.linear_regression import *"
__all__ = ["demo", "LinearReg_normal_equation", "LinearReg_statsmodels", "demo_regularization"]
