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
    print(counts)
    mapper = {0: 'Legitimate', 1: 'Fraud'}

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
    plt.title("Transaction Fraud Simulation (Extreme Imbalanced Data)")
    plt.show()

    # method 1: try training on the true distribution to see how well it generalizes
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)



def demo():
    """
    This function provides a demo of selected functions in this module.
    """

    _demo()
