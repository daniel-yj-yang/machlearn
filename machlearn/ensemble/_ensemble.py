# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingRegressor

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

    def __init__(self, n_trees = 100, n_features='sqrt', sample_size_factor=1.0, max_depth=10, impurity_measure='entropy', verbose=False):
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
        self.X_train = None
        self.y_train = None
        self.trees = []
        self.fitted = False
    
    def fit(self, X, y):
        ### init
        self.X_train = X
        self.y_train = y
        if type(self.X_train) in [pd.DataFrame, pd.Series]:
            self.X_train = self.X_train.to_numpy()
        if type(self.y_train) in [pd.DataFrame, pd.Series]:
            self.y_train = self.y_train.to_numpy()

        ### for X.col
        total_features_n = self.X_train.shape[1]

        if self.n_features is None:
            self.n_features_to_sample = total_features_n
        elif self.n_features == 'sqrt':
            self.n_features_to_sample = int(np.sqrt(total_features_n))
        elif self.n_features == 'log2':
            self.n_features_to_sample = int(np.log2(total_features_n))
        else:
            self.n_features_to_sample = self.n_features
        
        ### for X_train.row
        total_samples_n = self.X_train.shape[0]
        self.n_rows_to_sample = int(total_samples_n * self.sample_size_factor)

        ### train each tree
        np.random.seed(1)
        self.trees = [self.fit_a_single_decision_tree(tree_annotation=f"{i}") for i in range(self.n_trees)]

        self.fitted = True
        return self # return the fitted estimator

    def fit_a_single_decision_tree(self, tree_annotation=None):
        rows_indices     = list(np.random.permutation(self.X_train.shape[0])[:self.n_rows_to_sample])
        features_indices = list(np.random.permutation(self.X_train.shape[1])[:self.n_features_to_sample])
        this_DT = decision_tree_classifier_from_scratch(max_depth = self.max_depth, impurity_measure = self.impurity_measure, features_indices_actually_used = features_indices, annotation = tree_annotation, verbose = self.verbose)
        this_DT.fit( X = self.X_train[rows_indices,:], y = self.y_train[rows_indices] )
        return this_DT

    def predict(self, X_test):
        from scipy.stats import mode
        return mode([this_DT.predict(X_test) for this_DT in self.trees], axis=0).mode[0]

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
        return np.array([this_DT.score(X_test, y_test) for this_DT in self.trees])

    def print_debugging_info(self):
        if self.fitted:
            print(f"Number of features to sample (with replacement) from X to train each tree: {self.n_features_to_sample}")
            print(f"Number of rows to sample (with replacement) from X to train each tree: {self.n_rows_to_sample}")



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
    def __init__(self, n_trees = 100, sample_size_factor=1.0, max_depth=10, impurity_measure='entropy', verbose=False):
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

from ..logistic_regression import logistic_regression_classifier
from ..kNN import kNN_classifier 
from ..SVM import SVM_classifier

class boosting_classifier_from_scratch(object):
    """
    The idea of boosting: one is weak, together is strong, iterative training leads to the best model.
    Strategy: To improve the predictive power by training a sequence of weak models. Each additional weak model is to compensate the weaknesses of its predecessors.
    That is, learners are learned sequentially with early learners fitting simpler models to the data and then analyzing data for errors. Then, consecutive trees were fit to solve for net error from the prior tree.
    
    When an input is misclassified by a hypothesis, its weight is increased so that next hypothesis is more likely to classify it correctly. 
    
    Combining the whole set of trees at the end converts weak learners into better performing model.

    AdaBoost, Adaptive Boosting: at every step the sample distribution was adapted to put more weight on misclassified samples and less weight on correctly classified samples.

    --------------------

    Boosting vs. Random Forest:
        - Component Tree depth: Boosting is max_depth=1, RF is max_depth=full
        - Trees grown: Boosting is sequentially, RF is independently
        - Final votes: Boosting is weighted, RF is equal
    """
    def __init__(self, max_iter=50, verbose=False):
        self.max_iter = max_iter
        self.X_train = None
        self.y_train = None
        self.fitted = False
        self.verbose = verbose
        self.y_class0_value = None
        self.y_class1_value = None
        self.y_classes_value_conversion_dict = {}

    def fit(self, X, y):

        ### init
        self.X_train = X
        self.y_train = y
        if type(self.X_train) in [pd.DataFrame, pd.Series]:
            self.X_train = self.X_train.to_numpy()
        if type(self.y_train) in [pd.DataFrame, pd.Series]:
            self.y_train = self.y_train.to_numpy()
        total_samples_n = self.X_train.shape[0]

        ### check
        from collections import Counter
        y_train_unique_values_sorted_list = sorted(Counter(self.y_train).keys())
        if len(y_train_unique_values_sorted_list) != 2:
            raise ValueError('y train must be binary')
        self.y_class0_value = y_train_unique_values_sorted_list[0]
        self.y_class1_value = y_train_unique_values_sorted_list[1]
        self.y_classes_value_conversion_dict = {self.y_class0_value: -1, self.y_class1_value: 1}
        if [self.y_class0_value, self.y_class1_value] != [-1, 1]:  # Target values should be ±1
            self.y_train = np.where(self.y_train == self.y_class0_value, -1, 1)
        
        ### initialize
        # self.sample_weights_all_iter is a just collection of same weight on each iteration, but not used directly in computation of boosting
        self.all_iters_sample_weights = np.zeros(shape=(total_samples_n, self.max_iter)) # col_i = iteration(iter_i)
        self.weak_learners = np.zeros(shape=(self.max_iter,), dtype=object) # the minimum trees
        self.weak_learners_voting_weights = np.zeros(shape=(self.max_iter,))
        self.errors = np.zeros(shape=(self.max_iter,))

        # A. uniform weights for iteration(iter_i=0)
        self.all_iters_sample_weights[:, 0] = np.ones(shape=(total_samples_n,)) / total_samples_n 

        ### B. learning iterations
        for iter_i in range(self.max_iter):
            # 1. find a weak learner via a base estimator, which will be used to minimize curr_error
            curr_iter_same_weight = self.all_iters_sample_weights[:, iter_i]
            this_weak_learner = decision_tree_classifier_from_scratch(max_depth=1, verbose=self.verbose)
            #this_weak_learner = logistic_regression_classifier(C=1e9)
            #this_weak_learner = kNN_classifier() # TypeError: fit() got an unexpected keyword argument 'sample_weight'
            #this_weak_learner = SVM_classifier() # ValueError: ndarray is not C-contiguous
            this_weak_learner.fit(X=self.X_train, y=self.y_train, sample_weight=curr_iter_same_weight) # sample_weight=curr_iter_same_weight is the key here
            y_pred = this_weak_learner.predict(self.X_train)  
            curr_error = curr_iter_same_weight[y_pred != self.y_train].sum() # calculate error and weak_learner weight from weak learner prediction

            # 2. set a voting weight for this weak learner based on its accuracy; stronger learner can contribute more in the final voting
            this_weak_learner_voting_weight = 0.5 * np.log((1 - curr_error) / curr_error)

            # 3. increase weights of misclassified observations
            # this step is critical, as it will be used in the estimator in the next iteration
            sample_weight_next_iter = curr_iter_same_weight * np.exp(-this_weak_learner_voting_weight * self.y_train * y_pred)
            
            # 4. re-normalize sample weight, and update sample weights for the next iteration, until final iteration
            sample_weight_next_iter /= sample_weight_next_iter.sum()
            if iter_i+1 != self.max_iter: # no need to update sample_weights if reaching final iteration
                self.all_iters_sample_weights[:, iter_i+1] = sample_weight_next_iter

            # 5. save results of current iteration
            self.weak_learners[iter_i] = this_weak_learner
            self.weak_learners_voting_weights[iter_i] = this_weak_learner_voting_weight
            self.errors[iter_i] = curr_error

        self.fitted = True
        return self
    
    def predict(self, X_test):
        # C. the final predictions as the weighted majority vote of the weak learner's predictions
        weak_learners_y_preds = np.array([this_weak_learner.predict(X_test) for this_weak_learner in self.weak_learners])
        return np.sign(np.dot(self.weak_learners_voting_weights, weak_learners_y_preds)) ### Target values should be ±1

    def predict_proba(self, X_test):
        pass

    def score(self, X_test, y_test):
        if type(X_test) in [pd.DataFrame, pd.Series]:
            X_test = X_test.to_numpy()
        if type(y_test) in [pd.DataFrame, pd.Series]:
            y_test = y_test.to_numpy()
        from sklearn.metrics import confusion_matrix
        y_test = np.where(y_test == self.y_class0_value, -1, 1) ### Target values should be ±1
        cm = confusion_matrix(y_test, self.predict(X_test))
        accuracy = np.trace(cm) / np.sum(cm)
        return accuracy

    def score_of_individual_trees(self, X_test, y_test):
        y_test = np.where(y_test == self.y_class0_value, -1, 1) ### Target values should be ±1
        return np.array([this_weak_learner.score(X_test, y_test) for this_weak_learner in self.weak_learners])

    def print_debugging_info(self):
        if self.fitted:
            print(f"self.weak_learners_voting_weights = {self.weak_learners_voting_weights}")
            print(f"self.errors = {self.errors}")
            weak_learners_y_preds = np.array([this_weak_learner.predict(self.X_train) for this_weak_learner in self.weak_learners])
            print(f"To get at the prediction based on X_train, take the sign of the following: {np.dot(self.weak_learners_voting_weights, weak_learners_y_preds)}")


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


from ..decision_tree import decision_tree_regressor, decision_tree_regressor_from_scratch

class gradient_boosting_regressor_from_scratch(object):
    def __init__(self, max_iter=300, learning_rate=0.1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weak_learners = []
        self.loss_history = []

    def _y_pred(self):
        """
        θ = _y_pred()
        Use DecisionTreeRegressor()
        """
        pass

    def _cost_function(self, y, y_hat):
        """
        J(θ) = _cost_function()
        this is 0.5 * MSE 
        """
        return 0.5 * ((y - y_hat) ** 2).mean()

    def _gradient(self, y, y_hat):
        """
        ∂J(θ)/∂θ = _gradient()
        """
        return -1 * (y - y_hat)

    def fit(self, X, y):
        y_hat = np.array([y.mean()]*len(y)) # average as the starting point prediction
        self.y_hat0 = y_hat
        self.loss_history.append(self._cost_function(y, y_hat))
        for epoch_i in range(self.max_iter):
            residuals = -1 * self._gradient(y, y_hat)
            this_weak_learner = decision_tree_regressor_from_scratch(max_depth=1)
            this_weak_learner.fit(X, residuals)
            self.weak_learners.append(this_weak_learner)
            y_pred = this_weak_learner.predict(X)
            y_hat += self.learning_rate * y_pred
            self.loss_history.append(self._cost_function(y, y_hat))
            print(f"after epoch #{epoch_i:3d}: cost = {self._cost_function(y, y_hat):.3f}")
        return

    def predict(self, X_test):
        y_hat = np.array([self.y_hat0[0]]*len(X_test))
        for this_weak_learner in self.weak_learners:
            y_hat += self.learning_rate * this_weak_learner.predict(X_test)
        return y_hat

    def plot_loss_history(self):
        import matplotlib.pyplot as plt
        # construct a figure that plots the loss over time
        plt.figure(figsize=(5, 5))
        plt.plot(range(len(self.loss_history)), self.loss_history, label='GBM Training Loss')
        plt.legend(loc=1)
        plt.xlabel("Training Epoch #")
        plt.ylabel("Loss, J(θ)")
        plt.show()


def gradient_boosting_regressor(*args, **kwargs):
    return GradientBoostingRegressor(*args, **kwargs)


#######################################################################################################################################


def _demo(dataset):

    if dataset == 'boston':
        # regressor example
        GBM = gradient_boosting_regressor_from_scratch(max_iter=100)
        from ..datasets import public_dataset
        [X, y, df] = public_dataset('boston')
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
        GBM.fit(X_train, y_train)
        GBM.plot_loss_history()
        from ..model_evaluation import evaluate_continuous_prediction
        R_squared, RMSE = evaluate_continuous_prediction(y_test, GBM.predict(X_test))
        print(f"R_squared = {R_squared:.3f}, RMSE = {RMSE:.3f}")

    if dataset == 'Social_Network_Ads':
        from ..datasets import public_dataset
        data = public_dataset('Social_Network_Ads')
        X = data[['Age', 'EstimatedSalary']]
        y = data['Purchased']
        y_classes = ['not_purchased (y=0)', 'purchased (y=1)']
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

        for model_i, model in enumerate([boosting_classifier_from_scratch(), random_forest_classifier_from_scratch(max_depth=6)]):
            print(f"\n------------ model: {repr(model)} -------------\n")
            model.fit(X_train, y_train)
            model.print_debugging_info()
            print(f"Predicted probabilities: {model.predict_proba(X_test)}")
            print(f"Predicted label: {model.predict(X_test)}")
            print(f"Accuracy: {model.score(X_test,y_test)}")
            print(f"Accuracy of individual trees: {model.score_of_individual_trees(X_test,y_test)}\n")


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

        ###################################################
        
        print("\ngenerate n_samples=1000")        
        generate_data(n_samples=1000)
        ### this will take a very long time to complete when n_samples is big
        from ..decision_tree import decision_tree_classifier_from_scratch
        DT = decision_tree_classifier_from_scratch(max_depth=2)
        DT.fit(X_train,y_train)
        print(f"\nUse decision_tree_classifier_from_scratch(max_depth=2). Accuracy: {DT.score(X_test,y_test):.3f}")

        RF = random_forest_classifier_from_scratch(n_trees=10, max_depth=2)
        RF.fit(X_train,y_train)
        print(f"\nUse random_forest_classifier_from_scratch(n_trees=10, max_depth=2). Accuracy: {RF.score(X_test,y_test):.3f}")

        Boosting = boosting_classifier_from_scratch(max_iter=10)
        Boosting.fit(X_train, y_train)
        print(f"\nUse boosting_classifier_from_scratch(max_iter=10). Accuracy: {Boosting.score(X_test,y_test):.3f}")

        ###################################################

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
        - dataset:         A string. Possible values: "randomly_generated", "Social_Network_Ads", "boston"
    """

    available_datasets = ("randomly_generated", "Social_Network_Ads", "boston", )

    if dataset in available_datasets:
        return _demo(dataset = dataset)
    else:
        raise ValueError(f"dataset [{dataset}] is not defined")

#
# References
#
# Random Forest: 
# https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249
#
# AdaBoosting: 
# https://geoffruddock.com/adaboost-from-scratch-in-python/
# https://www.cs.toronto.edu/~mbrubake/teaching/C11/Handouts/AdaBoost.pdf
#
# Gradient Boosting: 
# https://towardsdatascience.com/gradient-boosting-in-python-from-scratch-4a3d9077367 (regressor)
# https://machinelearningmastery.com/gradient-boosting-machine-ensemble-in-python/
# https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/ (classifier)
#
# Logistic regression in ensemble:
# https://www.quora.com/Can-we-use-ensemble-methods-for-logistic-regression
# 
# Histogram-Based Gradient Boosting:
# https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
#
