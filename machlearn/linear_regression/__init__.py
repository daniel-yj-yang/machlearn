# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from ._linear_regression import demo, Linear_regression_normal_equation, Linear_regression, demo_regularization, Ridge_regression, Lasso_regression

# this is for "from <package_name>.linear_regression import *"
__all__ = ["demo", "Linear_regression_normal_equation", "Linear_regression", "demo_regularization", "Ridge_regression", "Lasso_regression"]
