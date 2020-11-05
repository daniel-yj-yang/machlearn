# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

import numpy as np
import pandas as pd

# Reduction in uncertainty = gain in information
#def Information_Gain(y, X):
#   IG(y, X) = Entropy(y) - Entropy(y | X)

# “purity” means how homogenized a group is.

#from numpy import ma # masked array

# A good read: https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8#:~:text=Entropy%20is%20a%20measure%20of,general%20is%20to%20reduce%20uncertainty.&text=This%20is%20called%20Information%20Gain,gained%20about%20Y%20from%20X.

def Entropy(splitted_sample=[]):
    """
    For example: [194, 106]
    Entropy is a measure of disorder/uncertainty (= low purity, a lack of dominant class)
    Entropy vs. Gini impurity: both of them involve p_j * p_j, but Entropy takes the form of S = k_b*ln(Ω)
    """

    import math

    denominator = sum(splitted_sample)
    if denominator == 0:
        return 0
    Entropy_index = 0
    for numerator_i in range(len(splitted_sample)):
        p_i = splitted_sample[numerator_i]/denominator
        if p_i != 0:
            Entropy_index -= p_i * math.log2(p_i)
        #Entropy_index -= p_i * ma.array(ma.log2(p_i)).filled(0) # handle log2(0) warning
    return Entropy_index


def Gini_impurity(splitted_sample=[]):
    """
    For example: [194, 106]
    note: summation( p_j * (1 - p_j) ) = 1 - summation( p_j^2 )
    """
    denominator = sum(splitted_sample)
    if denominator == 0:
        return 0
    Gini_index = 0
    for numerator_i in range(len(splitted_sample)):
        p_i = splitted_sample[numerator_i]/denominator
        # p_j * (1 - p_j) is the likelihood of misclassifying a new instance
        Gini_index += p_i * (1 - p_i)
    return Gini_index


def impurity_measure_with_sample_weight(y_class0_value=0, y_class1_value=1, y=[], sample_weight=None, impurity_func=Entropy):
    """
    For example: y=[0,1,1,0,0,1], sample_weight=[0.1, 0.1, 0.2, 0.1, 0.1, 0.1]
    """

    y_class0_sample_weighted_sum = 0
    y_class1_sample_weighted_sum = 0

    length_y = len(y)
    if length_y == 0:
        return 0

    if sample_weight is None:
        sample_weight = np.ones(shape=(length_y,))

    #print(f"{sample_weight}")

    if length_y != len(sample_weight):
        raise ValueError("unequal inputs of y[] and sample_weight[]")

    y_class0_sample_weighted_sum = np.sum(sample_weight[y == y_class0_value])
    y_class1_sample_weighted_sum = np.sum(sample_weight[y == y_class1_value])

    for i in range(length_y):
        if y[i] not in [y_class0_value, y_class1_value]:
            raise ValueError(f"y[{i}]=[{y[i]}] must be either y_class0_value=[{y_class0_value}] or y_class1_value=[{y_class1_value}]")

    return impurity_func(splitted_sample=[y_class0_sample_weighted_sum, y_class1_sample_weighted_sum])


def Impurity_plot():

    import matplotlib.pyplot as plt

    n = 200
    X = np.arange(0,n/2+1,1)
    Y = X[::-1]

    def plot(impurity_index, X, index_name="", title="", y_ref=None):
        plt.figure(figsize=(5, 5))
        plt.plot(X, impurity_index)
        plt.axhline(y=y_ref, color='r', linestyle='--')
        plt.title(f"{title}")
        plt.xlabel(f'n splitted as y=0 (total #sample = {n})')
        plt.ylabel(f"{index_name}")
        plt.ylim([0, 1.1])
        plt.show()

    measure = np.empty_like(X)
    for i in range(len(X)):
        measure[i] = Gini_impurity([X[i], Y[i]])
    plot(measure, X, index_name="Gini Impurity", title="Gini Impurity Plot (highest = 0.5 for two classes)", y_ref=0.5)

    measure = np.empty_like(X)
    for i in range(len(X)):
        measure[i] = Entropy([X[i], Y[i]])
    plot(measure, X, index_name="Entropy",       title="Entropy Plot (highest = 1.0 for two classes)",       y_ref=1.0)


def demo_metrics():
    """
    These generally measure the homogeneity of the target variable within the subsets.
    """
    Impurity_plot()


class decision_tree_classifier_node(object):
    def __init__(self, curr_depth=None, curr_impurity=None, curr_sample_size=None, curr_y_distribution={}, best_split_feature_i=None, best_x_cutoff_value=None, y_class0_value=0, y_class1_value=1):
        self.curr_depth = curr_depth
        self.curr_impurity = curr_impurity
        self.curr_sample_size = curr_sample_size
        self.curr_y_distribution = curr_y_distribution

        y_class0_n = curr_y_distribution.get(y_class0_value)
        y_class1_n = curr_y_distribution.get(y_class1_value)
        y_classes_count = len(curr_y_distribution)

        if y_classes_count == 0:
            self.y_dominant_class = None
            self.y_class1_prob = None

        if y_classes_count == 1:
            self.y_dominant_class = list(curr_y_distribution.keys())[0]
            if self.y_dominant_class == y_class0_value:
                self.y_class1_prob = 0
            elif self.y_dominant_class == y_class1_value:
                self.y_class1_prob = 1
            else:
                raise ValueError(f"unexpected y_dominant_class (should be either {y_class0_value} or {y_class1_value})")

        if y_classes_count == 2:
            # must be {y_class0_value: sample_size_associated_with_y=y_class0_value, y_class1_value: sample_size_associated_with_y=y_class1_value}
            self.y_class1_prob = y_class1_n / (y_class1_n + y_class0_n)
            if self.y_class1_prob >= 0.50:
                self.y_dominant_class = y_class1_value
            else:
                self.y_dominant_class = y_class0_value

        if y_classes_count > 2:
            raise ValueError('more than 2 classes in y detected')

        self.best_split_feature_i = best_split_feature_i
        self.best_x_cutoff_value = best_x_cutoff_value
        self.left = None
        self.right = None
    
    def to_dict(self):
        return {'curr_depth': self.curr_depth, 'curr_impurity': f"{self.curr_impurity:.3f}" if self.curr_impurity is not None else None, 'curr_sample_size': self.curr_sample_size, 'curr_y_distribution': self.curr_y_distribution, 'curr_dominant_y_class': self.y_dominant_class, 'curr_y_class1_prob': f"{self.y_class1_prob:.3f}" if self.y_class1_prob is not None else None, 'best_split_feature_i': self.best_split_feature_i if self.best_x_cutoff_value is not None else None, 'best_x_cutoff_value': f"{self.best_x_cutoff_value:.3f}" if self.best_x_cutoff_value is not None else None}


class decision_tree_classifier_from_scratch(object):
    """
    Shortcoming of DT:
    (1) biggest information gain locally, but globally; that is, there is always a better tree in the overall picture
    (2) we may be asking too trivial questions at the greater depths (e.g., > 5 or 6-depth)
    """

    def __init__(self, max_depth = 10, impurity_measure='entropy', features_indices_actually_used='all', annotation=None, verbose=False):
        """
        "features_indices_actually_used": limits the analysis on only these feature indices if not 'all'
            for example, if there are 30 features, then "features_indices_actually_used" = [2, 15] means that only the 3th and 16th features will be used for analysis
        """
        self.features_indices_actually_used = features_indices_actually_used  # limits the analysis on only these feature indices
        self.max_depth = max_depth
        self.verbose=verbose
        self.root_node = decision_tree_classifier_node()
        if impurity_measure not in ['entropy', 'gini_impurity']:
            raise ValueError('invalid impurity_measure value')
        self.impurity_measure = impurity_measure
        if self.impurity_measure == 'entropy':
            self.impurity_func = Entropy
        elif self.impurity_measure == 'gini_impurity':
            self.impurity_func = Gini_impurity
        self.annotation = annotation
        # default values
        self.y_class0_value = 0
        self.y_class1_value = 1

    def find_best_split_in_one_specific_feature(self, x, y_true, sample_weight=None):

        if type(x) in [pd.DataFrame, pd.Series]:
            x = x.to_numpy()

        if type(y_true) in [pd.DataFrame, pd.Series]:
            y_true = y_true.to_numpy()

        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y_true),))

        best_impurity = float('Inf')

        from sortedcontainers import SortedSet

        y_classes_values = list(SortedSet(y_true))
        if len(y_classes_values) != 2:
            raise ValueError("y must be binary")

        from collections import Counter

        #y_counts = Counter(y_true)
        #before_split_impurity = self.impurity_func([ y_counts[y_classes_values[0]], y_counts[y_classes_values[1]] ])
        before_split_impurity = impurity_measure_with_sample_weight(y_class0_value=self.y_class0_value, y_class1_value=self.y_class1_value, y=y_true, sample_weight=sample_weight, impurity_func=self.impurity_func)

        x_values_array = list(SortedSet(x))
        x_values_array_length = len(x_values_array)

        for value_i, this_x_value in enumerate(x_values_array):  # iterating through each value in this feature

            if value_i < (x_values_array_length-1):
                this_x_cutoff_value = (this_x_value + x_values_array[value_i+1]) / 2
            else:
                this_x_cutoff_value = this_x_value

            # let's say we split y on this value of x
            y_pred_left_node_list = y_true[x <= this_x_cutoff_value]
            y_pred_left_node_counts = Counter(y_pred_left_node_list)
            left_node_y_true_array  = [ y_pred_left_node_counts[y_classes_values[0]],  y_pred_left_node_counts[ y_classes_values[1]] ]
            #left_node_impurity  = self.impurity_func(left_node_y_true_array)
            left_node_impurity = impurity_measure_with_sample_weight(y_class0_value=self.y_class0_value, y_class1_value=self.y_class1_value, y=y_pred_left_node_list, sample_weight=sample_weight[x <= this_x_cutoff_value], impurity_func=self.impurity_func)
            left_node_n = sum(y_pred_left_node_counts.values())
            
            y_pred_right_node_list = y_true[x > this_x_cutoff_value]
            y_pred_right_node_counts = Counter(y_pred_right_node_list)
            right_node_y_true_array = [ y_pred_right_node_counts[y_classes_values[0]], y_pred_right_node_counts[y_classes_values[1]] ]
            #right_node_impurity = self.impurity_func(right_node_y_true_array)
            right_node_impurity = impurity_measure_with_sample_weight(y_class0_value=y_classes_values[0], y_class1_value=y_classes_values[1], y=y_pred_right_node_list, sample_weight=sample_weight[x > this_x_cutoff_value], impurity_func=self.impurity_func)
            right_node_n = sum(y_pred_right_node_counts.values())

            this_y_true_split_array = [left_node_y_true_array, right_node_y_true_array]

            total_n = left_node_n + right_node_n
            if total_n != len(y_true):
                raise ValueError("internal inconsistency")

            this_weighted_impurity = (left_node_impurity * left_node_n / total_n) + (right_node_impurity * right_node_n / total_n)
            this_information_gain = before_split_impurity - this_weighted_impurity

            if self.verbose:
                print(f"#{value_i:3d}: x_cutoff_value = {this_x_cutoff_value: .3f}, impurity = {this_weighted_impurity:.3f}, information_gain = {this_information_gain:.3f}, split_y_true_array = {this_y_true_split_array}")

            if this_weighted_impurity < best_impurity:
                best_impurity = this_weighted_impurity
                best_information_gain = this_information_gain
                best_x_cutoff_value = this_x_cutoff_value
                best_y_true_split_array = this_y_true_split_array

            if best_impurity == 0: # a perfect split was found
                break

        return best_x_cutoff_value, best_impurity, best_information_gain, best_y_true_split_array

    def find_best_split_across_all_features(self, X, y_true, sample_weight=None):

        if type(X) in [pd.DataFrame, pd.Series]:
            X = X.to_numpy()

        if type(y_true) in [pd.DataFrame, pd.Series]:
            y_true = y_true.to_numpy()

        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y_true),))

        best_impurity = float('Inf')

        n_samples = X.shape[0]        
        n_features = X.shape[1]

        if self.features_indices_actually_used == 'all':
            features_indices_actually_used = range(n_features)
        else:
            features_indices_actually_used = self.features_indices_actually_used

        for this_feature_i in features_indices_actually_used:
            x_cutoff_value, impurity, information_gain, y_true_split_array = self.find_best_split_in_one_specific_feature(x=X[:,this_feature_i], y_true=y_true, sample_weight=sample_weight)
            if self.verbose:
                print(f"feature # {this_feature_i: 2d}, x_cutoff_value = {x_cutoff_value: .3f}, impurity = {impurity:.3f}, information_gain = {information_gain:.3f}, y_true_split_array = {y_true_split_array}")
            if impurity == 0:  # a perfect split was found
                best_split_feature_i = this_feature_i
                best_x_cutoff_value, best_impurity, best_information_gain, best_y_true_split_array = x_cutoff_value, impurity, information_gain, y_true_split_array
                break
            elif impurity < best_impurity:
                best_split_feature_i = this_feature_i
                best_x_cutoff_value, best_impurity, best_information_gain, best_y_true_split_array = x_cutoff_value, impurity, information_gain, y_true_split_array

        return best_split_feature_i, best_x_cutoff_value, best_impurity, best_information_gain, best_y_true_split_array

    def fit(self, X, y, depth=0, sample_weight=None):

        y_true = y
        
        # init
        if depth == 0:
            from sortedcontainers import SortedSet
            y_classes_values = list(SortedSet(y_true))
            if len(y_classes_values) != 2:
                raise ValueError("y must be binary")
            else:
                self.y_class0_value = y_classes_values[0]
                self.y_class1_value = y_classes_values[1]

        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y_true),))

        from collections import Counter
        
        if self.verbose:
            print(f"depth={depth}")

        if type(X) in [pd.DataFrame, pd.Series]:
            X = X.to_numpy()

        if type(y_true) in [pd.DataFrame, pd.Series]:
            y_true = y_true.to_numpy()

        curr_sample_size = len(y_true)
        if curr_sample_size == 0:  # no data left to be split
            return None

        from sortedcontainers import SortedDict
        curr_y_distribution = SortedDict(Counter(y_true))

        #curr_impurity = self.impurity_func(list(Counter(y_true).values()))
        curr_impurity = impurity_measure_with_sample_weight(y_class0_value=self.y_class0_value, y_class1_value=self.y_class1_value, y=y_true, sample_weight=sample_weight, impurity_func=self.impurity_func)

        if curr_impurity == 0 or depth >= self.max_depth: # curr_impurity = 0 means already perfect, no need to split
            curr_node = decision_tree_classifier_node(curr_depth = depth, curr_impurity = curr_impurity, curr_sample_size = curr_sample_size, curr_y_distribution = curr_y_distribution, best_split_feature_i = None, best_x_cutoff_value = None, y_class0_value=self.y_class0_value, y_class1_value=self.y_class1_value)
            return curr_node

        best_split_feature_i, best_x_cutoff_value, best_impurity, best_information_gain, best_y_true_split_array = self.find_best_split_across_all_features(X=X, y_true=y_true, sample_weight=sample_weight)
        left_rows = X[:, best_split_feature_i] <= best_x_cutoff_value
        right_rows = X[:, best_split_feature_i] > best_x_cutoff_value
        X_left, X_right = X[left_rows], X[right_rows]
        y_true_left, y_true_right = y_true[left_rows], y_true[right_rows]
        sample_weight_left, sample_weight_right = sample_weight[left_rows], sample_weight[right_rows]
        curr_node = decision_tree_classifier_node(curr_depth = depth, curr_impurity = curr_impurity, curr_sample_size = curr_sample_size, curr_y_distribution = curr_y_distribution, best_split_feature_i = best_split_feature_i, best_x_cutoff_value = best_x_cutoff_value, y_class0_value=self.y_class0_value, y_class1_value=self.y_class1_value)
        # adding left and right children nodes into the node dict
        curr_node.left  = self.fit( X=X_left,  y=y_true_left,  sample_weight=sample_weight_left,  depth=depth+1)
        curr_node.right = self.fit( X=X_right, y=y_true_right, sample_weight=sample_weight_right, depth=depth+1)
        #print(parent_node)

        if depth == 0:
            self.root_node = curr_node
            return self # return the fitted estimator
        else:
            return curr_node

    def _order(self, curr_node, type="Inorder"):
        # Recursive travesal
        if curr_node:
            if type == "Inorder":
                return_dict = {}
                return_dict['left'] = self._order(curr_node.left, type=type)
                return_dict['curr'] = curr_node.to_dict()
                return_dict['right'] = self._order(curr_node.right, type=type)
                return return_dict
            if type == "Preorder":
                return_dict = {}
                return_dict['curr'] = curr_node.to_dict()
                return_dict['left'] = self._order(curr_node.left, type=type)
                return_dict['right'] = self._order(curr_node.right, type=type)
                return return_dict
            if type == "Postorder":
                return_dict = {}
                return_dict['left'] = self._order(curr_node.left, type=type)
                return_dict['right'] = self._order(curr_node.right, type=type)
                return_dict['curr'] = curr_node.to_dict()
                return return_dict

    def order(self, type="Inorder"):
        return self._order(curr_node=self.root_node, type=type)

    def _predict(self, one_X_row, proba=False):
        curr_node = self.root_node
        while curr_node.best_x_cutoff_value: # not a leaf node yet
            if one_X_row[ curr_node.best_split_feature_i ] <= curr_node.best_x_cutoff_value:
                curr_node = curr_node.left  # but what if curr_node.left = None?
            else:
                curr_node = curr_node.right # but what if curr_node.right = None?
        # arriving at a leaf node now
        if proba:
            return np.array([1-curr_node.y_class1_prob, curr_node.y_class1_prob])
        else:
            return np.array([curr_node.y_dominant_class])

    def predict(self, X, proba=False):
        if type(X) in [pd.DataFrame, pd.Series]:
            X = X.to_numpy()

        n_rows = X.shape[0]
        if proba:
            prediction = np.zeros(shape=(n_rows, 2))
        else:
            prediction = np.zeros(shape=(n_rows,))

        for this_row_i in range(n_rows):
            prediction[this_row_i] = self._predict(X[this_row_i,:], proba=proba)
        return prediction
     
    def predict_proba(self, X):
        return self.predict(X, proba=True)

    def score(self, X_test, y_test):
        if type(X_test) in [pd.DataFrame, pd.Series]:
            X_test = X_test.to_numpy()
        if type(y_test) in [pd.DataFrame, pd.Series]:
            y_test = y_test.to_numpy()
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, self.predict(X_test))
        accuracy = np.trace(cm) / np.sum(cm)
        return accuracy


def decision_tree_classifier(*args, **kwargs):
    """
    """
    return DecisionTreeClassifier(*args, **kwargs)


#######################################################################################################################################

class decision_tree_regressor_node(object):
    def __init__(self, X, y, subset_sample_indices, min_samples_leaf=5, max_depth=None, curr_depth=0):
        """
        min_samples_leaf: The minimum number of samples required to be at a leaf node.
        """
        self.X = X 
        self.y = y
        self.subset_sample_indices = subset_sample_indices
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.curr_depth = curr_depth
        self.n_samples = len(subset_sample_indices)
        self.n_features = X.shape[1]
        self.predicted_value = np.mean(y[subset_sample_indices]) # the decision (prediction) is based on the value the node holds.
        # best_after_split_purity_score, the lower the better, meaning more homogeneous elements within left and right nodes
        self.best_after_split_purity_score = float('inf') # the score indicates how effect the split was. It will be the weighted sum of the variance in the left and right nodes. leaf nodes do not have a score, thus set to infinity.
        if self.max_depth is None:
            self.find_best_feature_to_split()
        elif self.curr_depth < self.max_depth:
            self.find_best_feature_to_split()
        
    def find_best_feature_to_split(self):
        for feature_index in range(self.n_features):
            self.check_if_better_split_feature_and_its_value(feature_index)
        if self.is_leaf_node:
            return
        best_split_feature_x_series = self.best_split_feature_x_series
        best_split_left_node_series_indices  = np.nonzero(best_split_feature_x_series <= self.best_split_feature_value)[0]
        best_split_right_node_series_indices = np.nonzero(best_split_feature_x_series >  self.best_split_feature_value)[0]
        # recursively find all the left and right children nodes
        self.left  = decision_tree_regressor_node(X=self.X, y=self.y, subset_sample_indices=self.subset_sample_indices[best_split_left_node_series_indices],  min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, curr_depth=self.curr_depth+1)
        self.right = decision_tree_regressor_node(X=self.X, y=self.y, subset_sample_indices=self.subset_sample_indices[best_split_right_node_series_indices], min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, curr_depth=self.curr_depth+1)
        
    def check_if_better_split_feature_and_its_value(self, feature_index):
        """
        try to see if this feature and a certain value of the feature is a better split;
        criterion: a lower "weighted averages of the standard deviations", which is equivalent to minimizing RMSE.
        """      
        specific_x_series = self.X[self.subset_sample_indices, feature_index]

        for this_row in range(self.n_samples):
            specific_x_value_in_this_row = specific_x_series[this_row]
            left_node_series_indices  = specific_x_series <= specific_x_value_in_this_row # left_node  holds those in specific_x_series smaller than (or equal to) specific_x_value_in_this_row
            right_node_series_indices = specific_x_series >  specific_x_value_in_this_row # right_node holds those in specific_x_series larger than                specific_x_value_in_this_row
            n_samples_in_left_node  = left_node_series_indices.sum() # a shorthand way to count
            n_samples_in_right_node = right_node_series_indices.sum()
            # the leaf nodes must have at least n = "min_samples_left" samples
            if (n_samples_in_right_node < self.min_samples_leaf) or (n_samples_in_left_node < self.min_samples_leaf):
                continue

            # this_row is now a candidate as it satisfies the condition that the leaf nodes must have at least n = "min_samples_left" samples
            curr_after_split_purity_score = self.find_after_split_purity_score(left_node_series_indices, right_node_series_indices)
            if curr_after_split_purity_score < self.best_after_split_purity_score: 
                # a better split has been found
                self.best_split_feature_index = feature_index
                self.best_after_split_purity_score = curr_after_split_purity_score
                self.best_split_feature_value = specific_x_value_in_this_row
    
    def find_after_split_purity_score(self, left_node_indices, right_node_indices):
        """
        the weighted sum of the variance of y within left and right nodes
        """
        y_subset = self.y[self.subset_sample_indices]
        left_node_variance  = y_subset[left_node_indices].var()
        right_node_variance = y_subset[right_node_indices].var()
        n_samples_in_left_node  = left_node_indices.sum()
        n_samples_in_right_node = right_node_indices.sum()
        return (left_node_variance * n_samples_in_left_node) + (right_node_variance * n_samples_in_right_node)
                
    @property
    def best_split_feature_x_series(self):
        return self.X[self.subset_sample_indices, self.best_split_feature_index]
                
    @property
    def is_leaf_node(self):
        return self.best_after_split_purity_score == float('inf')                

    def predict(self, X_test):
        return np.array([self.predict_row(X_test_specific_row) for X_test_specific_row in X_test])

    def predict_row(self, X_test_specific_row):
        if self.is_leaf_node:
            return self.predicted_value
        child_node = self.left if X_test_specific_row[self.best_split_feature_index] <= self.best_split_feature_value else self.right
        return child_node.predict_row(X_test_specific_row)


class decision_tree_regressor_from_scratch(object):

    def __init__(self, min_samples_leaf=5, max_depth=None):
        """
        min_samples_leaf: The minimum number of samples required to be at a leaf node.
        """
        self.X_train = None
        self.y_train = None
        self.DT_root_node = None
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if type(self.X_train) in [pd.DataFrame, pd.Series]:
            self.X_train = self.X_train.to_numpy()
        if type(self.y_train) in [pd.DataFrame, pd.Series]:
            self.y_train = self.y_train.to_numpy()
        n_samples = len(self.y_train)
        self.DT_root_node = decision_tree_regressor_node(X=self.X_train, y=self.y_train, subset_sample_indices=np.arange(n_samples), min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth)
        return self
  
    def predict(self, X):
        X_test = X
        if type(X_test) in [pd.DataFrame, pd.Series]:
            X_test = X_test.to_numpy()
        return self.DT_root_node.predict(X_test)
        

def decision_tree_regressor(*args, **kwargs):
    """
    """
    return DecisionTreeRegressor(*args, **kwargs)


#######################################################################################################################################


def _demo(dataset="Social_Network_Ads", classifier_func="decision_tree"): # DT: decision_tree
    """
    classifier_func: "decision_tree" or "DT", "GBM", "AdaBoost", "bagging"
    """
    from ..datasets import public_dataset
    from ..ensemble import bagging_classifier, random_forest_classifier, adaptive_boosting_classifier, gradient_boosting_classifier

    if dataset == "iris":
        data = public_dataset(name="iris")
        y_classes = ['setosa', 'versicolor', 'virginica']
        X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
        y = data['target']

    if dataset == "Social_Network_Ads":
        data = public_dataset(name="Social_Network_Ads")
        print(f"{data.head()}\n")
        del data['User ID']
        # Recode the data: Gender as Male
        mapper = {'Male': 1, 'Female': 0}
        data['Male'] = data['Gender'].map(mapper)
        # pairplot
        import seaborn as sns
        sns.pairplot(data, hue="Purchased", markers=["o", "s"])
        import matplotlib.pyplot as plt
        plt.show()
        # X and y
        X = data[['Age', 'EstimatedSalary']] # dropping Male to simplify the analysis
        y = data['Purchased']
        y_classes = ['not_purchased (y=0)', 'purchased (y=1)']

    if dataset == "bank_note_authentication":
        data = public_dataset(name="bank_note_authentication")
        y_classes = ['genuine (y=0)', 'forged (y=1)']
        X = data[['variance', 'skewness', 'curtosis', 'entropy']]
        y = data['class']
        import seaborn as sns
        sns.pairplot(data, hue="class", markers=["o", "s"])

    from sklearn.model_selection import train_test_split, GridSearchCV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train) # fit to the data and then transform it
    # X_test = scaler.transform(X_test) # 	uses a previously computed mean and std to scale the data

    # create pipeline
    from sklearn.pipeline import Pipeline

    ########################################################################################################################
    if classifier_func in ["decision_tree", "DT"]:
        pipeline = Pipeline(steps=[('scaler', StandardScaler(with_mean=True, with_std=True)),
                                   ('classifier', decision_tree_classifier(max_depth=1, random_state=123)),  # default criterion = 'gini'
                                   ])

        # pipeline parameters to tune
        hyperparameters={
            'scaler__with_mean': [True],
            'scaler__with_std': [True],
            'classifier__criterion': ("gini", "entropy",),
            'classifier__max_depth': range(1, 15),
        }

        model_name = "Decision Tree"

    ########################################################################################################################
    if classifier_func == "bagging":
        pipeline = Pipeline(steps=[('scaler', StandardScaler(with_mean=True, with_std=True)),
                                   ('classifier', bagging_classifier(random_state=123)),
                                   ])

        # pipeline parameters to tune
        hyperparameters = {
            'scaler__with_mean': [True],
            'scaler__with_std': [True],
        }

        model_name = "Bagging"
        
    ########################################################################################################################
    if classifier_func in ["random_forest"]:
        pipeline = Pipeline(steps=[('scaler', StandardScaler(with_mean=True, with_std=True)),
                                   ('classifier', random_forest_classifier(max_depth=1, random_state=123)),  # default criterion = 'gini'
                                   ])

        # pipeline parameters to tune
        hyperparameters={
            'scaler__with_mean': [True],
            'scaler__with_std': [True],
            'classifier__criterion': ("gini", "entropy",),
            'classifier__max_depth': range(1, 10),
        }

        model_name = "Random Forest"

    ########################################################################################################################
    if classifier_func == "AdaBoost":
        pipeline = Pipeline(steps=[('scaler', StandardScaler(with_mean=True, with_std=True)),
                                   ('classifier', adaptive_boosting_classifier(random_state=123)),
                                   ])

        # pipeline parameters to tune
        hyperparameters = {
            'scaler__with_mean': [True],
            'scaler__with_std': [True],
        }

        model_name = "Adaptive Boosting"

    ########################################################################################################################
    if classifier_func == "GBM":
        pipeline = Pipeline(steps=[('scaler', StandardScaler(with_mean=True, with_std=True)),
                                   ('classifier', gradient_boosting_classifier(max_depth=1, random_state=123)),
                                   ])

        # pipeline parameters to tune
        hyperparameters = {
            'scaler__with_mean': [True],
            'scaler__with_std': [True],
            'classifier__criterion': ("friedman_mse", "mse", "mae"),
            'classifier__max_depth': range(1, 10),
        }

        model_name = "Gradient Boosting"

    ########################################################################################################################

    grid=GridSearchCV(
        pipeline,
        hyperparameters,  # parameters to tune via cross validation
        refit=True,       # fit using all data, on the best detected classifier
        n_jobs=-1,
        scoring='accuracy',
        cv=5,
    )

    # Training and predicting
    # classifier = DecisionTreeClassifier(max_depth=3) # default criterion = 'gini'
    # classifier = classifier.fit(X_train,y_train)
    classifier_grid=grid.fit(X_train, y_train)

    ########################################################################################################################
    if classifier_func in ["decision_tree", "DT", "random_forest", "GBM"]:
        criterion=classifier_grid.best_params_['classifier__criterion']
        max_depth=classifier_grid.best_params_['classifier__max_depth']
        print(
            f"Using a grid search and a {model_name} classifier, the best hyperparameters were found as following:\n"
            f"Step1: scaler: StandardScaler(with_mean={repr(classifier_grid.best_params_['scaler__with_mean'])}, with_std={repr(classifier_grid.best_params_['scaler__with_std'])});\n"
            f"Step2: classifier: {classifier_func}(criterion={repr(criterion)}, max_depth={repr(max_depth)}).\n")
        model_desc = f"{model_name} (criterion={repr(criterion)}, max_depth={repr(max_depth)})"

    if classifier_func in ["AdaBoost", "bagging"]:
        print(
            f"Using a grid search and a {model_name} classifier, the best hyperparameters were found as following:\n"
            f"Step1: scaler: StandardScaler(with_mean={repr(classifier_grid.best_params_['scaler__with_mean'])}, with_std={repr(classifier_grid.best_params_['scaler__with_std'])});\n"
            f"Step2: classifier: {classifier_func}().\n")
        model_desc = f"{model_name}"

    ########################################################################################################################

    y_pred=classifier_grid.predict(X_test)
    y_pred_score=classifier_grid.predict_proba(X_test)

    # Model evaluation
    from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves, visualize_classifier_decision_boundary_with_two_features
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, y_classes=y_classes)

    if dataset in ['Social_Network_Ads', 'bank_note_authentication']:
        plot_ROC_and_PR_curves(fitted_model=classifier_grid, X=X_test,
                            y_true=y_test, y_pred_score=y_pred_score[:, 1], y_pos_label=1, model_name=f"{model_name}")
    
    if dataset in ['Social_Network_Ads',]:
        visualize_classifier_decision_boundary_with_two_features(
            classifier_grid, X_train.to_numpy(), y_train.to_numpy(), y_classes, title=f"{model_desc} / training set", X1_lab='Age', X2_lab='Estimated Salary')
        visualize_classifier_decision_boundary_with_two_features(
            classifier_grid, X_test.to_numpy(),  y_test.to_numpy(),  y_classes, title=f"{model_desc} / testing set",  X1_lab='Age', X2_lab='Estimated Salary')

    # Plotting the tree
    if classifier_func in ["decision_tree", "DT"]:

        if dataset == 'iris':
            feature_cols = list(X.columns)
            #class_names = ['0', '1', '2']
            class_names = ['0=setosa', '1=versicolor', '2=virginica']

        if dataset == 'bank_note_authentication':
            feature_cols = list(X.columns)
            #class_names = ['0', '1']
            class_names = ['0=genuine', '1=forged']

        if dataset == 'Social_Network_Ads':
            feature_cols=['Age', 'EstimatedSalary']
            #class_names = ['0', '1']
            class_names = ['0=not purchased', '1=purchased']

        # Approach 1
        from dtreeviz.trees import dtreeviz
        viz = dtreeviz(classifier_grid.best_estimator_.steps[1][1], StandardScaler().fit_transform(X_train), y_train, target_name="target", feature_names=feature_cols, class_names=class_names)
        # print(type(viz)) # <class 'dtreeviz.trees.DTreeViz'>
        viz.view()

        # Approach 2
        from sklearn.tree import export_graphviz
        import io
        import pydotplus
        # from IPython.display import Image
        dot_data=io.StringIO()

        export_graphviz(classifier_grid.best_estimator_.steps[1][1], out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=feature_cols, class_names=class_names)

        graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
        image=graph.create_png()
        from PIL import Image
        Image.open(io.BytesIO(image)).show()


def demo(dataset="Social_Network_Ads", classifier_func="decision_tree"):
    """
    This function provides a demo of selected functions in this module.

    Required arguments:
        - dataset:         A string. Possible values: "Social_Network_Ads", "iris", "bank_note_authentication"
        - classifier_func: A string. Possible values: "decision_tree" or "DT", "random_forest", "bagging", "AdaBoost", "GBM"
    """

    available_datasets = ("Social_Network_Ads","iris","bank_note_authentication")
    available_classifier_functions = ("decision_tree", "DT", "random_forest", "bagging", "AdaBoost", "GBM",)

    if dataset in available_datasets and classifier_func in available_classifier_functions:
        return _demo(dataset = dataset, classifier_func = classifier_func)
    else:
        raise TypeError(f"either dataset [{dataset}] or classifier function [{classifier_func}] is not defined")


def demo_from_scratch(question_type="classification"):
    """
    "classification" or "regression"
    """
    if question_type not in ['classification', 'regression',]:
        raise ValueError('question_type should be either classification or regression')

    if question_type == "regression":
        from sklearn.datasets import load_boston
        boston = load_boston()
        X = pd.DataFrame(data=boston.data,columns=boston.feature_names)
        y = pd.DataFrame(data=boston.target,columns=['MEDV'])
        data = pd.concat([y, X], axis=1)
        print(f"{data.head()}\n")

        for model in [decision_tree_regressor_from_scratch(min_samples_leaf=5), decision_tree_regressor(min_samples_leaf=5)]:
            print(f"------ model: {repr(model)} ------")
            model.fit(X, y)
            y_pred = model.predict(X)
            from ..model_evaluation import evaluate_continuous_prediction
            R_squared, RMSE = evaluate_continuous_prediction(y, y_pred)
            print(f"R_squared = {R_squared:.3f}, RMSE = {RMSE:.3f}")
    
    if question_type == "classification":
        impurity_measure = 'entropy'  # impurity_measure: 'entropy' or 'gini_impurity'
        max_depth = 2

        from ..datasets import public_dataset
        data = public_dataset('Social_Network_Ads')
        X = data[['Age', 'EstimatedSalary']]
        y = data['Purchased']
        y_classes = ['not_purchased (y=0)', 'purchased (y=1)']
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

        DT_model = decision_tree_classifier_from_scratch(impurity_measure=impurity_measure, max_depth = max_depth)
        from sklearn.preprocessing import scale
        print(DT_model.find_best_split_in_one_specific_feature(x=scale(X_train['Age']), y_true=y_train))
        print(DT_model.find_best_split_across_all_features(X=scale(X_train), y_true=y_train))

        DT_model.fit(X_train, y_train)
        print(DT_model.order(type="Preorder"))
        print(f"\nAccuracy in predicting the target in the testing set: {DT_model.score(X_test, y_test)}")

        #print(DT_model.predict(X_train))
        #print(DT_model.predict_proba(X_train))
        from ..model_evaluation import plot_confusion_matrix, plot_ROC_curve, plot_ROC_and_PR_curves
        y_pred_score = DT_model.predict_proba(X_test)
        plot_confusion_matrix(y_true=y_test, y_pred=DT_model.predict(X_test), y_classes=y_classes)
        plot_ROC_curve(y_true=y_test, y_pred_score=y_pred_score[:,1])

#
# References
#
# decision tree:
#
# 1. Gini impurity: 
# https://towardsdatascience.com/gini-impurity-measure-dbd3878ead33
# http://www.learnbymarketing.com/481/decision-tree-flavors-gini-info-gain/
# https://towardsdatascience.com/gini-index-vs-information-entropy-7a7e4fed3fcb
# https://medium.com/analytics-steps/understanding-the-gini-index-and-information-gain-in-decision-trees-ab4720518ba8
#
# 2. Entropy: 
# https://www.geeksforgeeks.org/gini-impurity-and-entropy-in-decision-tree-ml/
# 
# 3. Gini vs. Entropy
# https://datascience.stackexchange.com/questions/10228/when-should-i-use-gini-impurity-as-opposed-to-information-gain-entropy
#
# 4. DT from scratch in Python: 
# https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea
# https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
#
# 5. Decision Tree Regressor in Python
# https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19
#

# References
#
# Ensemble methods:
#
# 1. Methods evolution and comparison
# https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator
# https://towardsdatascience.com/decision-tree-ensembles-bagging-and-boosting-266a8ba60fd9

