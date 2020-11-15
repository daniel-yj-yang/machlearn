# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

class classifier(object):
    def __init__(self):
        self._estimator_type = "classifier"
        self.classes_ = None
