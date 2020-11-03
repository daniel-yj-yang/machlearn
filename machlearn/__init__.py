# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from .__about__ import (
    __version__,
)

# this is for "from <package_name> import *"
__all__ = ["model_evaluation",
            "datasets",
           # supervised
           "kNN",
           "naive_bayes",
           "SVM",
           "decision_tree",
           "neural_network",
           "logistic_regression",
           "linear_regression",
           "DSA",
           "imbalanced_data",
           "decomposition",
           "ensemble",
           ]

# this was originally for _naive_bayes.py and is more widely applicable to other modules
from .datasets import public_dataset
import os
os.environ["NLTK_DATA"] = public_dataset("nltk_data_path")
os.environ["SCIKIT_LEARN_DATA"] = public_dataset("scikit_learn_data_path")
