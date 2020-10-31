# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from ._decision_tree import demo, demo_metrics, decision_tree_classifier

# this is for "from <package_name>.decision_tree import *"
__all__ = ["demo", "demo_metrics", "decision_tree_classifier", ]
