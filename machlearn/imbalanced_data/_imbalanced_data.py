# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

# reference: https://pypi.org/project/imbalanced-learn/

def _demo():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples = 10000, n_features = 20, n_classes = 2, weights=[.999,], random_state=26)

    from collections import Counter
    counts = Counter(y)
    # print(counts)
    y_classes = ['Legitimate', 'Fraud']
    #mapper = {0: 'Legitimate', 1: 'Fraud'}
    mapper = dict(zip([0, 1], y_classes))

    # bar chart of y
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 5))
    bars_x = [mapper[i] for i in list(counts.keys())]
    bars_y = list(counts.values())
    plt.bar(bars_x, bars_y, color='maroon', width=0.4)
    # Create labels
    labels = [f"n = {i}" for i in list(counts.values())]
    x_position = [0,1]
    for i in range(2):
        plt.text(x = x_position[i] - 0.07 , y = bars_y[i] + 50, s = labels[i], size = 12)
    # 
    plt.xlabel("Transaction Type")
    plt.ylabel("No. of Transactions")
    plt.title(f"Transaction Fraud Simulation (Extreme Imbalanced Data: {100 * bars_y[1] / sum(bars_y):.2f}% positive)")
    plt.show()

    # step 1: first try training on the true distribution to see how well it generalizes
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=26)

    from ..decision_tree import decision_tree_classifier
    classifier = decision_tree_classifier(max_depth=4, random_state=26).fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_pred_score = classifier.predict_proba(X_test)

    from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, y_classes=y_classes)
    plot_ROC_and_PR_curves(fitted_model=classifier, X=X_test, y_true=y_test, y_pred_score=y_pred_score[:, 1], y_pos_label=1, model_name=f"Decision Tree")

    from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_score[:,1])
    ap = average_precision_score(y_test, y_pred_score[:, 1])
    print(f"------------------------------------------------------------------------------------------")
    print(f"Step1: try training on the true distribution to see how well it generalizes.")
    print(f"\nIn classification, we care mostly about being able to make a positive prediction (y_pred=1), whether it's COVID-19 case or cancer.")
    print(f"\nHowever, because classes may be imbalanced (there are baseline probabilities where y_true=0), to evaluate positive predictions, there are two different curves that can answer different questions:\n")
    print(f"(a) In ROC curve, we want to see a random positive sample tends to be ranked higher (receives higher pred_score as positive) than a random negative sample - which can translate as the area under curve.")
    print(f"\n    p(y_pred = 1 | y_true = 1) vs. p(y_pred = 1 | y_true = 0)\n")
    print(f"    this takes the baseline probabilities of y_true into account and is thus independent of baseline class imbalance.\n")
    print(f"(b) In contrast, PR curve is particularly useful when the classes are very imbalanced, as we want to see not only the classifier returns a majority of all truly positive samples (high recall) but also when it makes a positive prediction, the prediction tends to be true.")
    print(f"\n    p(y_true = 1 | y_pred = 1) vs. p(y_pred = 1 | y_true = 1)\n")
    print(f"    because precision is a function of class imbalance, the area under PR curve is thus indicating how often a positive prediction from the classifier tends to be true, given the class imbalance and across all thresholds.")
    print(f"\nWith this in mind, the current classifier's performance is as follows:\n\n(a) area under ROC curve is [{roc_auc:.2f}] (how often a random positive sample is ranked higher than a random negative sample, regardless of class imbalance).\n\n(b) average precision is [{ap:.2f}] (how meaningful a positive prediction from the classifier is, given the class imbalance).\n")

    # step 2: downsample the majority class and upweight the majority class
    #


def demo():
    """
    This function provides a demo of selected functions in this module.
    """
    return _demo()
