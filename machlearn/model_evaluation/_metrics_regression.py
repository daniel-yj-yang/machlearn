# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import numpy as np

def SE(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    squared error
    """
    return np.square(y_true - y_pred)


def SSE(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    sum of squared error
    """
    return SE(y_true, y_pred).sum()


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    mean squared error
    Pros: penalizing large errors
    """
    return SE(y_true, y_pred).mean()


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    root mean squared error
    Pros: penalizing large errors
    """
    return MSE(y_true, y_pred) ** 0.5


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    mean absolute error
    Pros: easy to interpret
    """
    return np.absolute(y_true - y_pred).mean()


def demo_metrics_regression():
    y_true = np.random.rand(1, 100)
    y_pred = np.random.rand(1, 100)
    for func in [SSE, MSE, RMSE, MAE]:
        print(f"{repr(func)}: {func(y_true, y_pred):.3f}")

