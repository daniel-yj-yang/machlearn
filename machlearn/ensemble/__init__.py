# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from ._ensemble import demo
from ._ensemble import bagging_classifier, random_forest_classifier, adaptive_boosting_classifier, gradient_boosting_classifier, voting_classifier
from ._ensemble import bagging_classifier_from_scratch, random_forest_classifier_from_scratch, adaptive_boosting_classifier
from ._ensemble import gradient_boosting_regressor
from ._ensemble import gradient_boosting_regressor_from_scratch

# this is for "from <package_name>.ensemble import *"
__all__ = ["demo", 
           "bagging_classifier", 
           "random_forest_classifier", 
           "adaptive_boosting_classifier", 
           "gradient_boosting_classifier",
           "voting_classifier",
           "random_forest_classifier_from_scratch", 
           "bagging_classifier_from_scratch",
           "adaptive_boosting_classifier_from_scratch",
           "gradient_boosting_regressor",
           "gradient_boosting_regressor_from_scratch",]
