# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

class classifier(object):
    def __init__(self):
        # this is to work with sklearn, e.g., plot_precision_recall_curve
        self._estimator_type = "classifier"
        self.classes_ = [0,1,]
