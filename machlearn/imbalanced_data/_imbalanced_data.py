# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

# reference: https://pypi.org/project/imbalanced-learn/


def _demo(classifier_func= "logistic_regression"):
    
    available_classifiers = ['logistic_regression', 'decision_tree',] # these classifiers accept class_weight param
    if classifier_func not in available_classifiers:
        raise ValueError(f'provided classifier_func [classifier_func] has not been defined. available options: {available_classifiers}.')

    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=10000, n_features=30, n_redundant=2, n_classes=2, weights=[.995, ], flip_y=0, random_state=1, class_sep=0.80)
    y_classes = ['Legitimate', 'Fraud']

    def bar_chart(y):
        from collections import Counter
        counts = Counter(y)
        # print(counts)
        
        nonlocal y_classes
        #mapper = {0: 'Legitimate', 1: 'Fraud'}
        mapper = dict(zip([0, 1], y_classes))

        import math
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(7, 5))
        bars_x_axis = [mapper[i] for i in list(counts.keys())]
        bars_y_axis = list(counts.values())
        y_max = max(bars_y_axis)
        y_min = 0
        y_range = y_max - y_min
        plt.ylim([None, math.ceil(y_max+0.10*y_range)])
        plt.bar(bars_x_axis, bars_y_axis, color='maroon', width=0.4)
        # Create labels
        labels = [f"n = {i}" for i in list(counts.values())]
        x_position = [-0.11,0.92]
        for i in range(2):
            plt.text(x = x_position[i] , y = bars_y_axis[i] + y_range*0.02, s = labels[i], size = 12)
        # 
        plt.xlabel("Transaction Type")
        plt.ylabel("No. of Transactions")
        imbalance_pct = 100 * bars_y_axis[1] / sum(bars_y_axis)
        # https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data
        if imbalance_pct < 1:
            desc = 'Extreme Imbalanced Data: '
        elif imbalance_pct < 20:
            desc = 'Moderate Imbalanced Data: '
        elif imbalance_pct < 40:
            desc = 'Mild Imbalanced Data: '
        else:
            desc = ''
        plt.title(f"Transaction Fraud Simulation ({desc}{imbalance_pct:.2f}% positive)")
        plt.show()

    def run_classification(y, X, classifier_func, class_weight=None, sample_weight_dict=None):

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

        sample_weight=None
        if sample_weight_dict is not None:
            def sample_weight_func(x):
                return sample_weight_dict[x]
            sample_weight = np.vectorize(sample_weight_func)(y_train)
        print(f"\nmanually specified sample_weight: {sample_weight}")

        if classifier_func == "logistic_regression":
            from ..logistic_regression import logistic_regression_classifier
            classifier = logistic_regression_classifier(solver='liblinear', fit_intercept=True, max_iter=1e5, tol=1e-8, C=1e10, random_state=26, class_weight=class_weight)
            classifier.fit(X_train, y_train, sample_weight=sample_weight)
            model_name = 'Logistic Regression'

        if classifier_func == "decision_tree":
            from sklearn.pipeline import Pipeline
            from ..decision_tree import decision_tree_classifier
            pipeline = Pipeline(steps=[('classifier', decision_tree_classifier(random_state=1, class_weight=class_weight)),])
            hyperparameters = {
                'classifier__criterion': ("gini", "entropy",),
                'classifier__max_depth': range(1, 10),
            }
            from sklearn.model_selection import GridSearchCV
            grid = GridSearchCV(
                pipeline,
                hyperparameters,  # parameters to tune via cross validation
                refit=True,       # fit using all data, on the best detected classifier
                n_jobs=-1,
                scoring='accuracy',
                cv=5,
            )
            classifier_grid = grid.fit(X_train, y_train, classifier__sample_weight=sample_weight)
            criterion = classifier_grid.best_params_['classifier__criterion']
            max_depth = classifier_grid.best_params_['classifier__max_depth']
            classifier = classifier_grid
            #classifier = decision_tree_classifier(max_depth=6, random_state=1, class_weight = class_weight)
            #classifier.fit(X_train, y_train)
            model_name = f"DT ({criterion}, m.d.={max_depth})"
            
        y_pred = classifier.predict(X_test)
        y_pred_score = classifier.predict_proba(X_test)

        from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred, y_classes=y_classes)
        plot_ROC_and_PR_curves(fitted_model=classifier, X=X_test, y_true=y_test, y_pred_score=y_pred_score[:, 1], y_pos_label=1, model_name=f"{model_name}")

        from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_score[:,1])
        ap = average_precision_score(y_test, y_pred_score[:, 1])

        print(f"\nIn classification, we care mostly about being able to make a positive prediction (y_pred=1), whether it's COVID-19 case or cancer.")
        print(f"\nHowever, because classes may be imbalanced (there are baseline probabilities where y_true=0), to evaluate positive predictions, there are two different curves that can answer different questions:\n")
        print(f"(a) In ROC curve, we want to see a random positive sample tends to be ranked higher (receives higher pred_score as positive) than a random negative sample - which can translate as the area under curve.")
        print(f"\n    p(y_pred = 1 | y_true = 1) vs. p(y_pred = 1 | y_true = 0)\n")
        print(f"    this takes the baseline probabilities of y_true into account and is thus independent of baseline class imbalance.\n")
        print(f"(b) In contrast, PR curve is particularly useful when the classes are very imbalanced, as we want to see not only the classifier returns a majority of all truly positive samples (high recall) but also when it makes a positive prediction, the prediction tends to be true.")
        print(f"\n    p(y_true = 1 | y_pred = 1) vs. p(y_pred = 1 | y_true = 1)\n")
        print(f"    because precision is a function of class imbalance, the area under PR curve is thus indicating how often a positive prediction from the classifier tends to be true, given the class imbalance and across all thresholds.")
        print(f"\nWith this in mind, the current classifier's performance is as follows:\n\n(a) area under ROC curve is [{roc_auc:.2f}] (how often a random positive sample is ranked higher than a random negative sample, regardless of class imbalance).\n\n(b) average precision is [{ap:.2f}] (how meaningful a positive prediction from the classifier is, given the class imbalance).\n")

        return classifier

    # step 1: first try training on the true distribution to see how well it generalizes
    print(f"------------------------------------------------------------------------------------------")
    print(f"Step1: try training on the true distribution to see how well it generalizes.")
    bar_chart(y)
    classifier_step1 = run_classification(y, X, classifier_func=classifier_func, class_weight=None) # 1.

    # step 2: downsample the majority class (to match minority class), and then upweight the majority class
    y_original = y
    X_original = X

    classifier_step2a = []
    classifier_step2b = []

    # if 20,  before: class0 = 200 vs. class1 = 1, after: class0 = 10 vs. class1 = 1
    # if 200, before: class0 = 200 vs. class1 = 1, after: class0 = 1  vs. class1 = 1
    for i, down_sampling_factor in enumerate([20, 200]):
        y = y_original
        X = X_original

        print(f"------------------------------------------------------------------------------------------")
        print(f"Step2a: try downsampling (with a factor of {down_sampling_factor}) the majority class to see how well it generalizes.")

        import numpy as np
        # Indicies of each class' observations
        y_class0_idx = np.where(y == 0)[0]
        y_class1_idx = np.where(y == 1)[0]

        y_class0_n = len(y_class0_idx)
        y_class1_n = len(y_class1_idx)

        y_class0_n_downsampled = y_class0_n // down_sampling_factor

        np.random.seed(1)
        y_class0_idx_downsampled = np.random.choice(y_class0_idx, size=y_class0_n_downsampled, replace=False)
        y = np.hstack((y[y_class0_idx_downsampled], y[y_class1_idx]))
        X = np.vstack((X[y_class0_idx_downsampled], X[y_class1_idx]))

        bar_chart(y)
        classifier_step2a.append( run_classification(y, X, classifier_func=classifier_func, class_weight=None) )# 2a.

        print(f"------------------------------------------------------------------------------------------")
        print(f"Step2b: try upweighting the majority class to see how well it generalizes.\n")
        y_class0_weight = down_sampling_factor
        y_class1_weight = 1
        #total_classes_weights = y_class0_weight + y_class1_weight
        #y_class0_weight /= total_classes_weights
        #y_class1_weight /= total_classes_weights
        class_weight_dict = {0: y_class0_weight, 1: y_class1_weight}

        # sklearn's 'balanced' is to upweight the minority class, not the majority class
        from sklearn.utils import class_weight
        for class_weight_param in [None, 'balanced', class_weight_dict]:
            computed_class_weight = class_weight.compute_class_weight(class_weight=class_weight_param, classes = np.array([0,1]), y = y)
            print(f"sklearn computed class_weight for class_weight=[{class_weight_param}]: {computed_class_weight}")

        # sample_weight provides a more granular way to specify sample-specific weight than class weight
        # but if the whole class has the same weight, it is the same as class_weight
        class_weight_dict = None
        sample_weight_dict = {0: down_sampling_factor, 1: 1}

        print(f"\nmanually specified class_weight: {class_weight_dict}")
        classifier_step2b.append( run_classification(y, X, classifier_func=classifier_func, class_weight=class_weight_dict, sample_weight_dict=sample_weight_dict) ) # 2b.
        # Some references:
        # https://github.com/scikit-learn-contrib/imbalanced-learn
        # https://stackoverflow.com/questions/30972029/how-does-the-class-weight-parameter-in-scikit-learn-work
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        # https://imbalanced-learn.org/stable/auto_examples/index.html

        # https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
        # https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data
        # https://chrisalbon.com/machine_learning/preprocessing_structured_data/handling_imbalanced_classes_with_downsampling/

    return classifier_step1, classifier_step2a, classifier_step2b


def demo(classifier_func="logistic_regression"):
    """
    This function provides a demo of selected functions in this module.

    From simulation with decision_tree or logistic regression (these two accept class_weight param), it suggests that as expected, PR curve (but not ROC curve) can readily pick up the imbalance problem.
    To mitigate the problem, it is best to downsample the majority to match the minority class.
    
    While downsampling helps a lot, upweighting the majority class only helps when downsampling majority is still giving imbalanced data and not yet matching to the minority class.
    """
    return _demo(classifier_func = classifier_func)
