# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from ._linear_regression import demo, linear_regression_normal_equation, linear_regression, demo_regularization, ridge_regression, lasso_regression, linear_regression_torch
from ._linear_regression import linear_regression_sklearn, test_linear_regression_assumptions, demo_test_assumptions

# this is for "from <package_name>.linear_regression import *"
__all__ = ["demo", "linear_regression_normal_equation", "linear_regression", "demo_regularization", "ridge_regression", "lasso_regression", "linear_regression_torch",
           "linear_regression_sklearn", "test_linear_regression_assumptions", "demo_test_assumptions"]
