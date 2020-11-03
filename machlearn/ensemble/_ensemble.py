# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


#############################################################################
# improvement relationship:
#
# decision tree -> bagging -> random_forest -> boosting -> gradient boosting
#############################################################################

def bagging(*args, **kwargs):
    """
    an improvement to DT
    """
    return BaggingClassifier(*args, **kwargs)


def random_forest(*args, **kwargs):
    """
    an improvement to bagging
    """
    return RandomForestClassifier(*args, **kwargs)


def boosting(*args, **kwargs):
    """
    The idea of boosting: one is weak, together is strong, iterative training leads to the best model.
    Strategy: To improve the predictive power by training a sequence of weak models. Each additional weak model is to compensate the weaknesses of its predecessors.

    AdaBoost, Adaptive Boosting: at every step the sample distribution was adapted to put more weight on misclassified samples and less weight on correctly classified samples.
    """
    return AdaBoostClassifier(*args, **kwargs)


def gradient_boosting(*args, **kwargs):
    """
    The idea of boosting: one is weak, together is strong, iterative training leads to the best model.
    Strategy: To improve the predictive power by training a sequence of weak models. Each additional weak model is to compensate the weaknesses of its predecessors.

    GBM: Gradient Boosting Machines, including XGBOOST
    """
    return GradientBoostingClassifier(*args, **kwargs)



def _demo(dataset):
    pass



def demo(dataset="Social_Network_Ads"):
    """
    This function provides a demo of selected functions in this module.

    Required argument:
        - dataset:         A string. Possible values: "Social_Network_Ads"
    """

    available_datasets = ("Social_Network_Ads",)

    if dataset in available_datasets:
        return _demo(dataset = dataset)
    else:
        raise ValueError(f"dataset [{dataset}] is not defined")
