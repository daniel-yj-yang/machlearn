# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd

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

from ..decision_tree import decision_tree_classifier_from_scratch

class random_forest_classifier_from_scratch(object):
    """
    An extension of bagging. 
    
    In addition to taking the random subset of data, it also takes the random selection of features rather than using all features to grow trees.

    Pros: 
    1. As in bagging, it handles higher dimensionality data very well.
    2. As in bagging, it handles missing values and maintains accuracy for missing data.

    Cons:
    1. The average predictions from subset trees does not give precise values for the regression model.

    Two major components:
    1. Uncorrelated trees via random set of rows in X, y and random set of cols in X
        - Feature bagging: As average error of perfectly random errors is zero, we need "uncorrelated decision trees" via random subsets of features.
        - Bagging trees: a random set of rows
    2. Aggregation of the crappy trees
        In the case of regression, we can take the average (mean) of the prediction made by each tree.
        In the case of classification, we can take the majority (mode) of the class voted by each tree.
    """

    # some reference: https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249

    def __init__(self, n_trees = 100, n_features='sqrt', sample_size_factor=1.0, max_depth=10, impurity_measure='entropy', verbose=True):
        """
        n_features: this is where feature (X.col) bagging happens; the number of features sampled and passed onto to each tree. It can be:
            - 'sqrt': square root of total features #
            - 'log2': log base 2 of total features #
            - an interger
            - None: max_features
        
        sample_size_factor: this is where sample (X.row) bagging happens; it will draw sample_size_factor * X.shape[0] rows. max. = 1.0
        """
        self.n_trees = n_trees

        self.n_features = n_features
        self.n_features_to_sample = None

        if sample_size_factor > 1.0:
            raise ValueError("the max. sample_size_factor is capped at 1.0")
        self.sample_size_factor = sample_size_factor
        self.n_rows_to_sample = None

        self.max_depth = max_depth
        self.impurity_measure = impurity_measure
        self.verbose = verbose
        self.X = None
        self.y = None
        self.trees = []
    

    def fit(self, X_train, y_train):
        ### init
        self.X = X_train
        self.y = y_train
        if type(self.X) in [pd.DataFrame, pd.Series]:
            self.X = self.X.to_numpy()
        if type(self.y) in [pd.DataFrame, pd.Series]:
            self.y = self.y.to_numpy()

        ### for X.col
        total_features_n = self.X.shape[1]

        if self.n_features is None:
            self.n_features_to_sample = total_features_n
        elif self.n_features == 'sqrt':
            self.n_features_to_sample = int(np.sqrt(total_features_n))
        elif self.n_features == 'log2':
            self.n_features_to_sample = int(np.log2(total_features_n))
        else:
            self.n_features_to_sample = self.n_features
        
        if self.verbose:
            print(f"Number of features to sample (with replacement) from X to train each tree: {self.n_features_to_sample}")

        ### for X.row
        total_samples_n = self.X.shape[0]
        self.n_rows_to_sample = int(total_samples_n * self.sample_size_factor)

        if self.verbose:
            print(f"Number of rows to sample (with replacement) from X to train each tree: {self.n_rows_to_sample}")

        ### train each tree
        np.random.seed(1)
        self.trees = [self.fit_a_single_decision_tree(tree_annotation=f"{i}") for i in range(self.n_trees)]


    def fit_a_single_decision_tree(self, tree_annotation=None):
        rows_indices     = list(np.random.permutation(self.X.shape[0])[:self.n_rows_to_sample])
        features_indices = list(np.random.permutation(self.X.shape[1])[:self.n_features_to_sample])
        this_DT = decision_tree_classifier_from_scratch(max_depth = self.max_depth, impurity_measure = self.impurity_measure, features_indices_actually_used = features_indices, annotation = tree_annotation)
        this_DT.fit( X = self.X[rows_indices,:], y_true = self.y[rows_indices] )
        return this_DT

    
    def predict(self, X_test):
        from scipy.stats import mode
        return mode([this_DT.predict(X_test) for this_DT in self.trees], axis=0).mode[0,:,0]


    def predict_proba(self, X_test):
        return np.mean([this_DT.predict_proba(X_test) for this_DT in self.trees], axis=0)


    def score(self, X_test, y_test):
        if type(X_test) in [pd.DataFrame, pd.Series]:
            X_test = X_test.to_numpy()
        if type(y_test) in [pd.DataFrame, pd.Series]:
            y_test = y_test.to_numpy()
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, self.predict(X_test))
        accuracy = np.trace(cm) / np.sum(cm)
        return accuracy


    def score_of_individual_trees(self, X_test, y_test):
        return [this_DT.score(X_test, y_test) for this_DT in self.trees]


def random_forest_classifier(*args, **kwargs):
    """
    same as in random_forest_from_scratch
    """
    return RandomForestClassifier(*args, **kwargs)


#######################################################################################################################################


class bagging_classifier_from_scratch(random_forest_classifier_from_scratch):
    """
    Goal: to reduce the variance of a decision tree.
    To build several estimators independently and then to average their predictions.
    The combined estimator is usually better than any of the single base estimator because of reduced variance.

    Bagging stands for "B"ootstrap "Agg"regation.

    The idea is to create subsets of data, chosen randomly with replacement, from the training sample, and then to average all the predictions from different trees.
    Because of reduced variance, the averaged prediction is usually more robust than a single decision tree.
   """
    def __init__(self, n_trees = 100, sample_size_factor=1.0, max_depth=10, impurity_measure='entropy', verbose=True):
        """
            bagging is basically random_forest with "n_features=None"

            sample_size_factor: this is where sample (X.row) bagging happens; it will draw sample_size_factor * X.shape[0] rows. max. = 1.0
        """
        super().__init__(n_trees = n_trees, n_features=None, sample_size_factor = sample_size_factor, max_depth = max_depth, impurity_measure = impurity_measure, verbose = verbose)


def bagging_classifier(*args, **kwargs):
    """
    same as in bagging_from_scratch()
    """
    return BaggingClassifier(*args, **kwargs)


#######################################################################################################################################


class boosting_classifier_from_scratch(object):
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


def boosting_classifier(*args, **kwargs):
    """
    same as in boosting_from_scratch()
    """
    return AdaBoostClassifier(*args, **kwargs)


#######################################################################################################################################


class gradient_boosting_classifier_from_scratch(object):
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


def gradient_boosting_classifier(*args, **kwargs):
    """
    same as in gradient_boosting_from_scratch()
    """
    return GradientBoostingClassifier(*args, **kwargs)


#######################################################################################################################################


class voting_classifier_from_scratch(object):
    """
    To combine conceptually different machine learning classifiers and use a majority vote (hard vote) or the average predicted probabilities (soft vote) to predict the class labels. 
    Such a classifier can be useful for a set of equally well performing model in order to balance out their individual weaknesses.
    """
    def __init__(self):
        pass


def voting_classifier(*args, **kwargs):
    """
    same as in gradient_boosting_from_scratch()
    """
    return VotingClassifier(*args, **kwargs)
    

#######################################################################################################################################


def _demo(dataset):

    if dataset == 'Social_Network_Ads':
        from ..datasets import public_dataset
        data = public_dataset('Social_Network_Ads')
        X = data[['Age', 'EstimatedSalary']]
        y = data['Purchased']
        y_classes = ['not_purchased (y=0)', 'purchased (y=1)']
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

        RF_model = random_forest_classifier_from_scratch(max_depth=6)
        RF_model.fit(X_train, y_train)
        print(f"Predicted probabilities: {RF_model.predict_proba(X_test)}")
        print(f"Accuracy: {RF_model.score(X_test,y_test)}")
        print(f"Accuracy of individual trees: {RF_model.score_of_individual_trees(X_test,y_test)}")


    if dataset == "randomly_generated":

        print("Demo: Use an ensemble voting classifier (making predicitons by the majority vote or the averaged predicted probabilities), hoping to increase accuracy by cancelling out weakness in component models.")

        def generate_data(n_samples=1000):
            nonlocal X_train, X_test, y_train, y_test
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=n_samples, n_features=30, n_redundant=2, n_classes=2, weights=[.50, ], flip_y=0.02, random_state=1, class_sep=0.80)
            from collections import Counter
            y_counts = Counter(y)
            print(y_counts)

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1) # Setting ‘stratify’ to y makes our training split represent the proportion of each value in the y variable. 

        print("\ngenerate n_samples=1000")        
        generate_data(n_samples=1000)
        ### this will take a very long time to complete when n_samples is big
        from ..decision_tree import decision_tree_classifier_from_scratch
        DT = decision_tree_classifier_from_scratch(max_depth=2)
        DT.fit(X_train,y_train)
        print(f"\nUse decision_tree_classifier_from_scratch(). Accuracy: {DT.score(X_test,y_test):.3f}")

        print("\ngenerate n_samples=10000")
        generate_data(n_samples=10000)
        print("\ntune hyperparameters:")

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

        # model_3: decision tree
        from ..decision_tree import decision_tree_classifier
        model_decision_tree = decision_tree_classifier(random_state=1)
        hyperparameters_decision_tree = {"max_depth": range(1, 10)}
        grid_decision_tree = GridSearchCV(model_decision_tree, hyperparameters_decision_tree, cv=5)
        grid_decision_tree.fit(X_train, y_train)
        print(f"- best_params of decision tree: {grid_decision_tree.best_params_}")        

        # model_4: random forest
        model_random_forest = random_forest_classifier(random_state=1)
        hyperparameters_random_forest = {"n_estimators": [100, 200, 300, ]} # The number of trees in the forest
        grid_random_forest = GridSearchCV(model_random_forest, hyperparameters_random_forest, cv=5)
        grid_random_forest.fit(X_train, y_train)
        print(f"- best_params of random forest: {grid_random_forest.best_params_}")

        print("\nAccuracy:")
        print(f"- kNN: {grid_kNN.best_estimator_.score(X_test, y_test)}")
        print(f"- logistic regression: {model_log_reg.score(X_test, y_test)}")
        print(f"- decision tree: {grid_decision_tree.best_estimator_.score(X_test, y_test)}")
        print(f"- random forest: {grid_random_forest.best_estimator_.score(X_test, y_test)}")

        # ensemble
        estimator_list = [("kNN", grid_kNN.best_estimator_), ("log_reg", model_log_reg), ("DT", grid_decision_tree.best_estimator_), ("random_forest", grid_random_forest.best_estimator_)]
        ensemble_classifier = voting_classifier(estimator_list, voting = "hard") # make predicitons by majority vote of the class
        ensemble_classifier.fit(X_train, y_train)
        print(f"- ensemble (hard voting, making predicitons by the majority vote of the class label): {ensemble_classifier.score(X_test, y_test)}")

        ensemble_classifier = voting_classifier(estimator_list, voting = "soft") # make predicitons by majority vote of the class probabilites
        ensemble_classifier.fit(X_train, y_train)
        print(f"- ensemble (soft voting, making predicitons by the averaged class probabilities): {ensemble_classifier.score(X_test, y_test)}")

        # Output:
        # 
        # Counter({0: 5013, 1: 4987})
        #
        # Hyperparameters:
        # - best_params of kNN: {'n_neighbors': 27}
        # = best_params of decision tree: {'max_depth': 6}
        # - best_params of random forest: {'n_estimators': 200}
        #
        # Accuracy:
        # - kNN: 0.81
        # - logistic regression: 0.83
        # - decision tree: 0.9252
        # - random forest: 0.9312
        # - ensemble (hard voting, making predicitons by the majority vote of the class label): 0.8784
        # - ensemble (soft voting, making predicitons by the averaged class probabilities): 0.9268

        print("\nThe results suggest that voting is NOT guaranteed to provide better accuracy, as it is based on the majority label or the average predicted probabilities.")
        print("Rather, a voting classifier is more useful for a set of EQUALLY well performing model as it can balance out their individual weaknesses.")
        

def demo(dataset="randomly_generated"):
    """
    This function provides a demo of selected functions in this module.

    Required argument:
        - dataset:         A string. Possible values: "randomly_generated", "Social_Network_Ads"
    """

    available_datasets = ("randomly_generated", "Social_Network_Ads",)

    if dataset in available_datasets:
        return _demo(dataset = dataset)
    else:
        raise ValueError(f"dataset [{dataset}] is not defined")
