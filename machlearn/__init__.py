# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from .__about__ import (
    __version__,
)

# this is for "from <package_name> import *"
__all__ = ["SVM",
           "datasets",
           "decision_tree",
           "kNN",
           "model_evaluation",
           "naive_bayes",
           "neural_network",
           ]

# this was originally for _naive_bayes.py and is more widely applicable to other modules
from .datasets import public_dataset
import os
os.environ["NLTK_DATA"] = public_dataset("nltk_data_path")
os.environ["SCIKIT_LEARN_DATA"] = public_dataset("scikit_learn_data_path")
