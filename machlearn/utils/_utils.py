# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import pandas as pd
import numpy as np
from scipy.sparse import csr


def convert_to_numpy_ndarray(X):
    type_X = type(X)
    if type_X == np.ndarray:
        return X
    if type_X in [pd.DataFrame, pd.Series]:
        return X.to_numpy()
    if type_X == csr.csr_matrix:
        return X.toarray()
    if type_X == list:
        return np.array(X)
    if X is None:
        return np.array([None])
    raise TypeError(f"Unknown type of X: {type_X}")


def convert_to_list(X):
    type_X = type(X)
    if type_X == list:
        return X
    if type_X in [pd.Series,]:
        return X.to_list()
    if X is None:
        return [X]
    raise TypeError(f"Unknown type of X: {type_X}")


def demo():
    """
    This function provides a demo of selected functions in this module.
    """
    X = pd.DataFrame([[1,2,3],[4,5,6]])
    convert_to_numpy_ndarray(X)
    X = pd.Series([1,2,3,4,5,6])
    convert_to_numpy_ndarray(X)
    convert_to_list(X)
