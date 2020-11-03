# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier

# Reference:
# https://scikit-learn.org/stable/modules/ensemble.html#

#############################################################################
# improvement relationship:
#
# decision tree -> averaging methods: bagging, random_forest 
# decision tree -> boosting methods: boosting, gradient boosting
#############################################################################

#######################################################################################################################################


class bagging_from_scratch(object):
    """
    Goal: to reduce the variance of a decision tree.
    To build several estimators independently and then to average their predictions.
    The combined estimator is usually better than any of the single base estimator because of reduced variance.

    Bagging stands for Bootstrap Aggregation.

    The idea is to create subsets of data, chosen randomly with replacement, from the training sample, and then to average all the predictions from different trees.
    Because of reduced variance, the averaged prediction is usually more robust than a single decision tree.
    """
    def __init__(self):
        pass


def bagging(*args, **kwargs):
    """
    same as in bagging_from_scratch()
    """
    return BaggingClassifier(*args, **kwargs)


#######################################################################################################################################


class random_forest_from_scratch(object):
    """
    An extension of bagging. 
    
    In addition to taking the random subset of data, it also takes the random selection of features rather than using all features to grow trees.

    Pros: 
    1. As in bagging, it handles higher dimensionality data very well.
    2. As in bagging, it handles missing values and maintains accuracy for missing data.

    Cons:
    1. The average predictions from subset trees does not give precise values for the regression model.
    """
    def __init__(self):
        pass


def random_forest(*args, **kwargs):
    """
    same as in random_forest_from_scratch
    """
    return RandomForestClassifier(*args, **kwargs)


#######################################################################################################################################


class boosting_from_scratch(object):
    """
    The idea of boosting: one is weak, together is strong, iterative training leads to the best model.
    Strategy: To improve the predictive power by training a sequence of weak models. Each additional weak model is to compensate the weaknesses of its predecessors.
    That is, learners are learned sequentially with early learners fitting simpler models to the data and then analyzing data for errors. Then, consecutive trees were fit to solve for net error from the prior tree.
    
    When an input is misclassified by a hypothesis, its weight is increased so that next hypothesis is more likely to classify it correctly. 
    
    Combining the whole set of trees at the end converts weak learners into better performing model.

    AdaBoost, Adaptive Boosting: at every step the sample distribution was adapted to put more weight on misclassified samples and less weight on correctly classified samples.
    """
    def __init__(self):
        pass


def boosting(*args, **kwargs):
    """
    same as in boosting_from_scratch()
    """
    return AdaBoostClassifier(*args, **kwargs)


#######################################################################################################################################


class gradient_boosting_from_scratch(object):
    """
    The idea of boosting: one is weak, together is strong, iterative training leads to the best model.
    Strategy: To improve the predictive power by training a sequence of weak models. Each additional weak model is to compensate the weaknesses of its predecessors.

    Gradient Descent + Boosting

    Pros:
    1. Supports differentiable loss function.
    2. Works well with interactions.

    Cons: 
    1. Prone to over-fitting.
    2. Careful tuning of hyperparameters required

    GBM: Gradient Boosting Machines, including XGBOOST
    """
    def __init__(self):
        pass


def gradient_boosting(*args, **kwargs):
    """
    same as in gradient_boosting_from_scratch()
    """
    return GradientBoostingClassifier(*args, **kwargs)


#######################################################################################################################################


class voting_from_scratch(object):
    def __init__(self):
        pass


def voting(*args, **kwargs):
    return VotingClassifier(*args, **kwargs)
    

#######################################################################################################################################


def _demo(dataset):

    import numpy as np

    if dataset == "randomly_generated":

        print("Demo: Use an ensemble voting classifier (make predicitons by majority vote), hoping to increase accuracy.")

        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=10000, n_features=30, n_redundant=2, n_classes=2, weights=[.50, ], flip_y=0.02, random_state=1, class_sep=0.80)
        from collections import Counter
        y_counts = Counter(y)
        print(y_counts)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1) # Setting ‘stratify’ to y makes our training split represent the proportion of each value in the y variable. 
        
        print("\nHyperparameters:")

        # model_1: kNN
        from sklearn.model_selection import GridSearchCV
        from ..kNN import kNN_classifier
        model_kNN = kNN_classifier()
        hyperparams_kNN = {'n_neighbors': range(25, 30)}
        grid_kNN = GridSearchCV(model_kNN, hyperparams_kNN, cv=5)
        grid_kNN.fit(X_train, y_train)
        print(f"- best_params of kNN: {grid_kNN.best_params_}")

        # model_2: logistic regression
        # no tuning of hyperparameters required for logistic regression
        from ..logistic_regression import logistic_regression_classifier
        model_log_reg = logistic_regression_classifier()
        model_log_reg.fit(X_train, y_train)

        # model_3: random forest
        model_random_forest = random_forest(random_state=1)
        hyperparameters_random_forest = {"n_estimators": [200, 300, 400, ]}
        grid_random_forest = GridSearchCV(model_random_forest, hyperparameters_random_forest, cv=5)
        grid_random_forest.fit(X_train, y_train)
        print(f"- best_params of random forest: {grid_random_forest.best_params_}")

        print("\nAccuracy:")
        print(f"- kNN: {grid_kNN.best_estimator_.score(X_test, y_test)}")
        print(f"- logistic regression: {model_log_reg.score(X_test, y_test)}")
        print(f"- random forest: {grid_random_forest.best_estimator_.score(X_test, y_test)}")

        # ensemble
        estimator_list = [("kNN", grid_kNN.best_estimator_), ("log_reg", model_log_reg), ("random_forest", grid_random_forest.best_estimator_)]
        ensemble_classifier = voting(estimator_list) # make predicitons by majority vote
        ensemble_classifier.fit(X_train, y_train)
        print(f"- ensemble (voting, make predicitons by majority vote): {ensemble_classifier.score(X_test, y_test)}")

        # Output:
        #
        # Hyperparameters:
        # - best_params of kNN: {'n_neighbors': 27}
        # - best_params of random forest: {'n_estimators': 300}
        #
        # Accuracy:
        # - kNN: 0.81
        # - logistic regression: 0.83
        # - random forest: 0.9304
        # - ensemble (voting, make predicitons by majority vote): 0.8668
        #

        print("The results suggest that voting is NOT guaranteed to provide better accuracy, as it is based on the majority")
        

def demo(dataset="randomly_generated"):
    """
    This function provides a demo of selected functions in this module.

    Required argument:
        - dataset:         A string. Possible values: "randomly_generated"
    """

    available_datasets = ("randomly_generated",)

    if dataset in available_datasets:
        return _demo(dataset = dataset)
    else:
        raise ValueError(f"dataset [{dataset}] is not defined")
