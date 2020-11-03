# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from ._ensemble import demo, bagging, random_forest, boosting, gradient_boosting

# this is for "from <package_name>.ensemble import *"
__all__ = ["demo", "bagging", "random_forest", "boosting", "gradient_boosting", ]
