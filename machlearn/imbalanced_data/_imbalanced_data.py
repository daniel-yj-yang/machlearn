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
        plt.text(x = x_position[i] - 0.06 , y = bars_y[i] + 50, s = labels[i], size = 12)
    # 
    plt.xlabel("Transaction Type")
    plt.ylabel("No. of Transactions")
    plt.title(f"Transaction Fraud Simulation (Extreme Imbalanced Data: {100 * bars_y[1] / sum(bars_y):.2f}% positive)")
    plt.show()

    # solution 1: first try training on the true distribution to see how well it generalizes
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=26)

    from ..decision_tree import decision_tree_classifier
    classifier = decision_tree_classifier(max_depth=4, random_state=26).fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_pred_score = classifier.predict_proba(X_test)

    from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, y_classes=y_classes)
    plot_ROC_and_PR_curves(fitted_model=classifier, X=X_test, y_true=y_test, y_pred_score=y_pred_score[:, 1], y_pos_label=1, model_name=f"Decision Tree")

    # solution 2:


def demo():
    """
    This function provides a demo of selected functions in this module.
    """
    return _demo()
