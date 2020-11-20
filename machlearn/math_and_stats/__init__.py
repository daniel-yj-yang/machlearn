# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from ._stats import stats_demo, demo_probability, demo_distance, probability, distance
from ._math import math_demo

# this is for "from <package_name>.math_and_stats import *"
__all__ = ["stats_demo", "demo_probability", "demo_distance", "probability", "distance", "math_demo",]
