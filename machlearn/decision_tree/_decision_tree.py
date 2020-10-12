# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier


def boosting(type = "GBM"):
    """
    Idea: one is weak, together is strong, iterative training leads to the best model.
    Strategy: To improve the predictive power by training a sequence of weak models. Each additional weak model is to compensate the weaknesses of its predecessors.

    type: (1) GBM = Gradient Boosting Machines, including XGBOOST;
          (2) Adaptive = AdaBoost

    Details:
    (2) AdaBoost: at every step the sample distribution was adapted to put more weight on misclassified samples and less weight on correctly classified samples.
    """
    #return GradientBoostingClassifier(*args, **kwargs)
    return AdaBoostClassifier(*args, **kwargs)


def demo():
    """
    This function provides a demo of selected functions in this module.

    Required arguments:
        None
    """
