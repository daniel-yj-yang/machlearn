# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd

# Reduction in uncertainty = gain in information
#def Information_Gain(y, X):
#   IG(y, X) = Entropy(y) - Entropy(y | X)

# “purity” means how homogenized a group is.

from numpy import ma # masked array

# A good read: https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8#:~:text=Entropy%20is%20a%20measure%20of,general%20is%20to%20reduce%20uncertainty.&text=This%20is%20called%20Information%20Gain,gained%20about%20Y%20from%20X.
def Entropy(splitted_sample=[]):
    """
    For example: [194, 106]
    Entropy is a measure of disorder/uncertainty (= low purity, a lack of dominant class)
    Entropy vs. Gini impurity: both of them involve p_j * p_j, but Entropy takes the form of S = k_b*ln(Ω)
    """
    denominator = sum(splitted_sample)
    if denominator == 0:
        return 0
    Entropy_index = 0
    for numerator_i in range(len(splitted_sample)):
        p_i = splitted_sample[numerator_i]/denominator
        Entropy_index -= p_i * ma.array(ma.log2(p_i)).filled(0) # handle log2(0) warning
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

class decision_tree_node(object):
    def __init__(self, curr_depth=None, curr_impurity=None, curr_sample_size=None, curr_y_distribution={}, best_split_feature_i=None, best_x_cutoff_value=None):
        self.curr_depth = curr_depth
        self.curr_impurity = curr_impurity
        self.curr_sample_size = curr_sample_size
        self.curr_y_distribution = curr_y_distribution

        y_class0_n = curr_y_distribution.get(0)
        y_class1_n = curr_y_distribution.get(1)
        y_classes_count = len(curr_y_distribution)

        if y_classes_count == 0:
            self.y_dominant_class = None
            self.y_class1_prob = None

        if y_classes_count == 1:
            self.y_dominant_class = list(curr_y_distribution.keys())[0]
            if self.y_dominant_class == 0:
                self.y_class1_prob = 0
            elif self.y_dominant_class == 1:
                self.y_class1_prob = 1
            else:
                raise ValueError('unexpected y_dominant_class (should be either 0 or 1)')

        if y_classes_count == 2:
            # must be {0: sample_size_associated_with_y=0, 1: sample_size_associated_with_y=1}
            self.y_class1_prob = y_class1_n / (y_class1_n + y_class0_n)
            if self.y_class1_prob >= 0.50:
                self.y_dominant_class = 1
            else:
                self.y_dominant_class = 0

        if y_classes_count > 2:
            raise ValueError('more than 2 classes in y detected')

        self.best_split_feature_i = best_split_feature_i
        self.best_x_cutoff_value = best_x_cutoff_value
        self.left = None
        self.right = None
    
    def to_dict(self):
        return {'curr_depth': self.curr_depth, 'curr_impurity': f"{self.curr_impurity:.3f}", 'curr_sample_size': self.curr_sample_size, 'curr_y_distribution': self.curr_y_distribution, 'curr_dominant_y_class': self.y_dominant_class, 'curr_y_class1_prob': f"{self.y_class1_prob:.3f}" if self.y_class1_prob is not None else None, 'best_split_feature_i': self.best_split_feature_i if self.best_x_cutoff_value is not None else None, 'best_x_cutoff_value': f"{self.best_x_cutoff_value:.3f}" if self.best_x_cutoff_value is not None else None}


class decision_tree_classifier_from_scratch(object):

    def __init__(self, max_depth = 10, impurity_measure='entropy'):
        self.max_depth = max_depth
        self.verbose = False
        self.root_node = decision_tree_node()
        if impurity_measure not in ['entropy', 'gini_impurity']:
            raise ValueError('invalid impurity_measure value')
        self.impurity_measure = impurity_measure
        if self.impurity_measure == 'entropy':
            self.impurity_func = Entropy
        elif self.impurity_measure == 'gini_impurity':
            self.impurity_func = Gini_impurity

    def find_best_split_in_one_specific_feature(self, x, y_true):

        if type(x) == pd.DataFrame:
            x = x.to_numpy()

        if type(y_true) == pd.DataFrame:
            y_true = y_true.to_numpy()

        best_impurity = float('Inf')

        from sortedcontainers import SortedSet

        y_classes_values = list(SortedSet(y_true))
        if len(y_classes_values) != 2:
            raise ValueError("y must be binary")

        from collections import Counter

        y_counts = Counter(y_true)
        before_split_impurity = self.impurity_func([ y_counts[y_classes_values[0]], y_counts[y_classes_values[1]] ])

        x_values_array = list(SortedSet(x))
        x_values_array_length = len(x_values_array)

        for value_i, this_x_value in enumerate(x_values_array):  # iterating through each value in this feature

            if value_i < (x_values_array_length-1):
                this_x_cutoff_value = (this_x_value + x_values_array[value_i+1]) / 2
            else:
                this_x_cutoff_value = this_x_value

            # let's say we split y on this value of x
            y_pred_left_node_counts  = Counter(y_true[x <= this_x_cutoff_value])
            left_node_y_true_array  = [ y_pred_left_node_counts[y_classes_values[0]],  y_pred_left_node_counts[ y_classes_values[1]] ]
            left_node_impurity  = self.impurity_func(left_node_y_true_array)
            left_node_n = sum(y_pred_left_node_counts.values())
            
            y_pred_right_node_counts = Counter(y_true[x >  this_x_cutoff_value])
            right_node_y_true_array = [ y_pred_right_node_counts[y_classes_values[0]], y_pred_right_node_counts[y_classes_values[1]] ]
            right_node_impurity = self.impurity_func(right_node_y_true_array)
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


    def find_best_split_across_all_features(self, X, y_true):

        if type(X) == pd.DataFrame:
            X = X.to_numpy()

        if type(y_true) == pd.DataFrame:
            y_true = y_true.to_numpy()

        best_impurity = float('Inf')

        n_samples = X.shape[0]        
        n_features = X.shape[1]

        for feature_i in range(n_features):
            x_cutoff_value, impurity, information_gain, y_true_split_array = self.find_best_split_in_one_specific_feature(X[:,feature_i], y_true)
            if self.verbose:
                print(f"feature # {feature_i: 2d}, x_cutoff_value = {x_cutoff_value: .3f}, impurity = {impurity:.3f}, information_gain = {information_gain:.3f}, y_true_split_array = {y_true_split_array}")
            if impurity == 0:  # a perfect split was found
                best_split_feature_i = feature_i
                best_x_cutoff_value, best_impurity, best_information_gain, best_y_true_split_array = x_cutoff_value, impurity, information_gain, y_true_split_array
                break
            elif impurity < best_impurity:
                best_split_feature_i = feature_i
                best_x_cutoff_value, best_impurity, best_information_gain, best_y_true_split_array = x_cutoff_value, impurity, information_gain, y_true_split_array

        return best_split_feature_i, best_x_cutoff_value, best_impurity, best_information_gain, best_y_true_split_array

    def fit(self, X, y_true, depth=0):

        from collections import Counter
        
        if self.verbose:
            print(f"depth={depth}")

        if type(X) == pd.DataFrame:
            X = X.to_numpy()

        if type(y_true) == pd.DataFrame:
            y_true = y_true.to_numpy()

        curr_sample_size = len(y_true)
        if curr_sample_size == 0:  # no data left to be split
            return None

        from sortedcontainers import SortedDict
        curr_y_distribution = SortedDict(Counter(y_true))

        curr_impurity = self.impurity_func(list(Counter(y_true).values()))

        if curr_impurity == 0 or depth == self.max_depth: # curr_impurity = 0 means already perfect, no need to split
            curr_node = decision_tree_node(curr_depth = depth, curr_impurity = curr_impurity, curr_sample_size = curr_sample_size, curr_y_distribution = curr_y_distribution, best_split_feature_i = None, best_x_cutoff_value = None)
            return curr_node

        best_split_feature_i, best_x_cutoff_value, best_impurity, best_information_gain, best_y_true_split_array = self.find_best_split_across_all_features(X, y_true)
        left_rows = X[:, best_split_feature_i] <= best_x_cutoff_value
        right_rows = X[:, best_split_feature_i] > best_x_cutoff_value
        X_left, X_right = X[left_rows], X[right_rows]
        y_true_left, y_true_right = y_true[left_rows], y_true[right_rows]
        curr_node = decision_tree_node(curr_depth = depth, curr_impurity = curr_impurity, curr_sample_size = curr_sample_size, curr_y_distribution = curr_y_distribution, best_split_feature_i = best_split_feature_i, best_x_cutoff_value = best_x_cutoff_value)
        # adding left and right children nodes into the node dict
        curr_node.left  = self.fit( X_left,  y_true_left,  depth+1)
        curr_node.right = self.fit( X_right, y_true_right, depth+1)
        #print(parent_node)

        if depth == 0:
            self.root_node = curr_node
            return self
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


    def predict(self, X, proba=False):
        if type(X) == pd.DataFrame:
            X = X.to_numpy()

        n_rows = X.shape[0]
        if proba:
            prediction = np.zeros(shape=(n_rows, 2))
        else:
            prediction = np.zeros(shape=(n_rows, 1))
        for this_row_i in range(n_rows):
            prediction[this_row_i] = self._predict(X[this_row_i,:], proba=proba)
        return prediction

        

    def predict_proba(self, X):
        return self.predict(X, proba=True)


    def _predict(self, one_X_row, proba=False):
        curr_node = self.root_node
        while curr_node.best_x_cutoff_value: # not a leaf node yet
            if one_X_row[ curr_node.best_split_feature_i ] <= curr_node.best_x_cutoff_value:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
        # arriving at a leaf node now
        if proba:
            return np.array([1-curr_node.y_class1_prob, curr_node.y_class1_prob])
        else:
            return np.array([curr_node.y_dominant_class])


def decision_tree_classifier(*args, **kwargs):
    return DecisionTreeClassifier(*args, **kwargs)


def random_forest(*args, **kwargs):
    return RandomForestClassifier(*args, **kwargs)


def bagging(*args, **kwargs):
    return BaggingClassifier(*args, **kwargs)


def AdaBoost(*args, **kwargs):
    """
    The idea of boosting: one is weak, together is strong, iterative training leads to the best model.
    Strategy: To improve the predictive power by training a sequence of weak models. Each additional weak model is to compensate the weaknesses of its predecessors.

    AdaBoost, Adaptive Boosting: at every step the sample distribution was adapted to put more weight on misclassified samples and less weight on correctly classified samples.
    """
    return AdaBoostClassifier(*args, **kwargs)


def GBM(*args, **kwargs):
    """
    The idea of boosting: one is weak, together is strong, iterative training leads to the best model.
    Strategy: To improve the predictive power by training a sequence of weak models. Each additional weak model is to compensate the weaknesses of its predecessors.

    GBM: Gradient Boosting Machines, including XGBOOST
    """
    return GradientBoostingClassifier(*args, **kwargs)



def _demo(dataset="Social_Network_Ads", classifier_func="decision_tree"): # DT: decision_tree
    """
    classifier_func: "decision_tree" or "DT", "GBM", "AdaBoost", "bagging"
    """
    from ..datasets import public_dataset

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
    if classifier_func in ["random_forest"]:
        pipeline = Pipeline(steps=[('scaler', StandardScaler(with_mean=True, with_std=True)),
                                   ('classifier', random_forest(max_depth=1, random_state=123)),  # default criterion = 'gini'
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
    if classifier_func == "bagging":
        pipeline = Pipeline(steps=[('scaler', StandardScaler(with_mean=True, with_std=True)),
                                   ('classifier', bagging(random_state=123)),
                                   ])

        # pipeline parameters to tune
        hyperparameters = {
            'scaler__with_mean': [True],
            'scaler__with_std': [True],
        }

        model_name = "Bagging"

    ########################################################################################################################
    if classifier_func == "AdaBoost":
        pipeline = Pipeline(steps=[('scaler', StandardScaler(with_mean=True, with_std=True)),
                                   ('classifier', AdaBoost(random_state=123)),
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
                                   ('classifier', GBM(max_depth=1, random_state=123)),
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


def demo_DT_from_scratch(data="Social_Network_Ads", impurity_measure='entropy', max_depth=2):
    """
    impurity_measure: 'entropy' or 'gini_impurity'
    """
    
    from ..datasets import public_dataset
    data = public_dataset('Social_Network_Ads')
    X = data[['Age', 'EstimatedSalary']]
    y = data['Purchased']
    y_classes = ['not_purchased (y=0)', 'purchased (y=1)']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    DT_model = decision_tree_classifier_from_scratch(impurity_measure=impurity_measure, max_depth = max_depth)
    from sklearn.preprocessing import scale
    print(DT_model.find_best_split_in_one_specific_feature(scale(X_train['Age']), y_train))
    print(DT_model.find_best_split_across_all_features(scale(X_train), y_train))

    DT_model.fit(X_train, y_train)
    print(DT_model.order(type="Preorder"))
    #print(DT_model.predict(X_train))
    #print(DT_model.predict_proba(X_train))
    from ..model_evaluation import plot_confusion_matrix, plot_ROC_curve, plot_ROC_and_PR_curves
    y_pred_score = DT_model.predict_proba(X_test)
    plot_confusion_matrix(y_true=y_test, y_pred=DT_model.predict(X_test), y_classes=y_classes)
    plot_ROC_curve(y_true=y_test, y_pred_score=y_pred_score[:,1])
