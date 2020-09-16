# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["plot_confusion_matrix", "plot_ROC_curve",
           "plot_PR_curve", "plot_ROC_and_PR_curves",
           "test"]

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, average_precision_score, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from .naive_bayes import naive_bayes_Gaussian


__font_size__ = 18


def plot_confusion_matrix(cm,
                          y_classes=['y=0', 'y=1'],
                          figsize=(9, 9)):
    """
    This function generates an interpretable plot of confusion matrix, along with key statistics.

    Arguments:
        - cm_ndarray: A numpy.ndarray, the output from sklearn.metrics.confusion_matrix
        - y_classes:  A list, the y_classes to be displayed
        - figsize:    A tuple, the figure size. reference: plt.rcParams.get('figure.figsize')
    """

    if cm.shape == (2, 2):
        cell_label = ['True Negative', 'False Positive',
                      'False Negative', 'True Positive']
    else:
        cell_label = ['' for i in range(cm.size)]

    cell_count = ["{0:0.0f}".format(val) for val in cm.flatten()]

    cell_percentage = ["{0:.2%}".format(val)
                       for val in cm.flatten()/np.sum(cm)]

    cell_text = [f"{a}\n{b}\n{c}".strip() for a, b, c in zip(
        cell_label, cell_count, cell_percentage)]
    cell_text = np.asarray(cell_text).reshape(cm.shape)

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
        stats_text = "\n\nAccuracy(higher TP and TN) = (TP+TN)/Total = {:0.3f}\nF1 Score(lower FP and FN) = TP/(TP+0.5*(FP+FN)) = {:0.3f}\n\nRecall/sensitivity/TPR = 1-β = p($y_{{pred}}$=1 | $y_{{true}}$=1) = {:0.3f}\nFPR = α = p($y_{{pred}}$=1 | $y_{{true}}$=0) = {:0.3f}\n\nPrecision = 1-FDR = p($y_{{true}}$=1 | $y_{{pred}}$=1) = {:0.3f}".format(
            accuracy, f1_score, recall, alpha, precision)
    else:
        stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)

    old_font_size = plt.rcParams.get('font.size')
    plt.rcParams.update({'font.size': __font_size__})

    # Using seaborn to generate the visualization
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm,
                annot=cell_text,
                fmt="",
                cmap='Blues',  # or 'binary' # http://matplotlib.org/examples/color/colormaps_reference.html
                cbar=False,
                xticklabels=y_classes,
                linewidths=1.0,
                linecolor='black')

    ax.set_yticklabels(y_classes,
                       va='center',
                       rotation=90,
                       position=(0, 0.28))

    plt.ylabel('$y_{true}$')
    plt.xlabel('$y_{pred}$' + stats_text)

    fig.tight_layout()
    plt.show()

    plt.rcParams.update({'font.size': old_font_size})


def plot_ROC_curve(y_true,
                   y_pred_score,
                   y_pos_label=1,
                   figsize=(7, 6),
                   model_name='Binary Classifier'):
    """
    This function plots the ROC (Receiver operating characteristic) curve, along with statistics.

    Arguments:
        - y_true:       The labels could be {0,1}
        - y_pred_score: The probability estimates of the positive class
        - y_pos_label:        The label of the positive class (default = 1)
        - figsize:            A tuple, the figure size. reference: plt.rcParams.get('figure.figsize')
        - model_name:         A string
    """

    fpr, tpr, thresholds = roc_curve(
        y_true=y_true, y_score=y_pred_score, pos_label=y_pos_label)
    auc = roc_auc_score(y_true, y_pred_score)
    # auc = np.trapz(tpr,fpr) # alternatively

    old_font_size = plt.rcParams.get('font.size')
    plt.rcParams.update({'font.size': __font_size__})

    fig = plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f"{model_name} (AUC = {auc:0.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate = p($y_{pred}$=1 | $y_{true}$=0)')
    plt.ylabel('True Positive Rate = p($y_{pred}$=1 | $y_{true}$=1)')
    plt.title('ROC Curve')

    fig.tight_layout()
    plt.show()

    plt.rcParams.update({'font.size': old_font_size})


def plot_PR_curve(fitted_model,
                  X,
                  y_true,
                  y_pred_score,
                  y_pos_label=1,
                  figsize=(7, 6),
                  model_name='Binary Classifier'):
    """
    This function plots the precision-recall curve, along with statistics.

    Arguments:
        - fitted_model: A fitted classifier instance
        - X:            A matrix of m_samples x n_features
        - y_true:       The labels could be {0,1}
        - y_pred_score: The probability estimates of the positive class
        - y_pos_label:  The label of the positive class (default = 1)
        - figsize:      A tuple, the figure size. reference: plt.rcParams.get('figure.figsize')
        - model_name:   A string
    """

    AP = average_precision_score(
        y_true=y_true, y_score=y_pred_score, pos_label=y_pos_label, average='macro')

    old_font_size = plt.rcParams.get('font.size')
    plt.rcParams.update({'font.size': __font_size__})

    fig, ax = plt.subplots(figsize=figsize)

    plot_precision_recall_curve(estimator=fitted_model, X=X, y=y_true,
                                response_method='predict_proba', ax=ax, label=f"{model_name} (AP = {AP:0.2f})")
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall = p($y_{pred}$=1 | $y_{true}$=1)')
    ax.set_ylabel('Precision = p($y_{true}$=1 | $y_{pred}$=1)')

    fig.tight_layout()
    plt.show()

    plt.rcParams.update({'font.size': old_font_size})


def plot_ROC_and_PR_curves(fitted_model,
                           X,
                           y_true,
                           y_pred_score,
                           y_pos_label=1,
                           figsize=(8, 7),
                           model_name='Binary Classifier'):
    """
    This function plots both the ROC and the precision-recall curves, along with statistics.

    Arguments:
        - fitted_model: A fitted classifier instance
        - X:            A matrix of m_samples x n_features
        - y_true:       The labels could be {0,1}
        - y_pred_score: The probability estimates of the positive class
        - y_pos_label:  The label of the positive class (default = 1)
        - figsize:      A tuple, the figure size. reference: plt.rcParams.get('figure.figsize')
        - model_name:   A string
    """
    plot_ROC_curve(y_true=y_true, y_pred_score=y_pred_score,
                   y_pos_label=y_pos_label, figsize=figsize, model_name=model_name)

    plot_PR_curve(fitted_model=fitted_model, X=X,
                  y_true=y_true, y_pred_score=y_pred_score, y_pos_label=y_pos_label, figsize=figsize, model_name=model_name)


def test():
    """
    This function tests all the plotting functions in this module.
    """
    X, y = datasets.make_classification(
        n_samples=1000, n_features=30, n_classes=2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123)
    model = naive_bayes_Gaussian().fit(X_train, y_train)
    y_pred_score = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    plot_confusion_matrix(confusion_matrix(y_test, y_pred))
    plot_ROC_and_PR_curves(fitted_model=model, X=X_test, y_true=y_test,
                           y_pred_score=y_pred_score[:, 1], model_name='Gaussian NB')
