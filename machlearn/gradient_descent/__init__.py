# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from ._gradient_descent import demo, batch_gradient_descent, logistic_regression_BGD_classifier

# this is for "from <package_name>.gradient_descent import *"
__all__ = ["demo", "batch_gradient_descent", "logistic_regression_BGD_classifier", ]
