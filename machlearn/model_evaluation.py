import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm,
                          y_classes=['y=0', 'y=1'],
                          figsize=(12, 10)):
    """
    This function produces a more human-interpretable plot of confusion matrix along with key statistics

    Arguments:
    - cm:        A numpy.ndarray, the output from sklearn.metrics.confusion_matrix
    - y_classes: A list, the y_classes to be displayed
    - figsize:   A tuple, the figure size. reference: plt.rcParams.get('figure.figsize')
    """

    if cm.shape == (2, 2):
        cell_label = ['True Negative\n', 'False Positive\n',
                      'False Negative\n', 'True Positive\n']
    else:
        cell_label = ['' for i in range(cm.size)]

    cell_count = ["{0:0.0f}\n".format(val) for val in cm.flatten()]

    cell_percentage = ["{0:.2%}".format(val)
                       for val in cm.flatten()/np.sum(cm)]

    cell_text = [f"{x1}{x2}{x3}".strip() for x1, x2, x3 in zip(
        cell_label, cell_count, cell_percentage)]
    cell_text = np.asarray(cell_text).reshape(cm.shape[0], cm.shape[1])

    # Key statistics
    accuracy = np.trace(cm) / np.sum(cm)

    if cm.shape == (2, 2):  # Metrics for a 2x2 confusion matrix
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        # or, 2*precision*recall / (precision + recall)
        f1_score = TP / (TP + 0.5*(FP+FN))
        alpha = FP / (TN+FP)
        stats_text = "\n\nAccuracy(higher TP and TN) = (TP+TN)/Total = {: 0.3f}\nF1 Score(lower FP and FN) = TP/(TP+0.5*(FP+FN)) = {: 0.3f}\n\nRecall/sensitivity/TPR/power(1-β) = p(y_pred=1 | y_actual=1) = {: 0.3f}\nFPR/α = p(y_pred=1 | y_actual=0) = {: 0.3f}\n\nPrecision(1-FDR) = p(y_actual=1 | y_pred=1) = {: 0.3f}".format(
            accuracy, f1_score, recall, alpha, precision)
    else:
        stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)

    old_font_size = plt.rcParams.get('font.size')
    plt.rcParams.update({'font.size': 20})

    # Using seaborn to generate the visualization
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm,
                annot=cell_text,
                fmt="",
                cmap='Blues',  # or 'binary' # http://matplotlib.org/examples/color/colormaps_reference.html
                cbar=False,
                xticklabels=y_classes,
                linewidths=0.5,
                linecolor='black')

    ax.set_yticklabels(y_classes,
                       va='center',
                       rotation=90,
                       position=(0, 0.28))

    plt.ylabel('y_actual')
    plt.xlabel('y_predicted' + stats_text)

    fig.tight_layout()
    plt.show()

    plt.rcParams.update({'font.size': old_font_size})
