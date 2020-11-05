# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause


__font_size__ = 18

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, average_precision_score, plot_precision_recall_curve, classification_report


from sklearn.model_selection import KFold

def test_for_multicollinearity(X):
    """
    X: pd_DataFrame
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [VIF(X.values, i) for i in range(len(X.columns))]
    print(f"If Variance Inflation Factor (VIF) > 5, the 2+ numbers are collinear of each other and may be reduced to fewer using PCA.\n{vif_data}")


class K_Fold_CV(KFold):
    def __init__(self, k=5):
        super().__init__(n_splits=k, shuffle=True, random_state=123)


def demo_CV():
    # some ideas are from: https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f
    from sklearn import datasets
    X, y = datasets.make_classification(n_samples=5000, n_features=30, n_classes=2, class_sep=0.8, random_state=123)
    cv = K_Fold_CV()
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver='liblinear', fit_intercept=True, max_iter=1000, tol=1e-9, C=1e9)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    from numpy import mean
    from numpy import std
    print(f"Accuracy: mean={mean(scores):.3f} (std={std(scores):.3f})")


def evaluate_continuous_prediction(y_true, y_pred):
    if type(y_true) in [pd.DataFrame, pd.Series]:
        y_true = y_true.to_numpy()
    if type(y_pred) in [pd.DataFrame, pd.Series]:
        y_pred = y_pred.to_numpy()
    from sklearn.metrics import r2_score, mean_squared_error
    return r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred)**0.5
    #from statistics import mean
    #SSE = sum((y_true - y_pred) ** 2)
    #RMSE = ( SSE / len(y_true) ) ** 0.5
    #SST = sum((y_true - mean(y_true)) ** 2)
    #R_squared = 1 - (SSE/SST)
    #return RMSE, R_squared





def visualize_classifier_decision_boundary_with_two_features(classifier, X, y, y_classes, title='Classifier', X1_lab='X1', X2_lab='X2', figsize=(8, 7)):
    """
    # reference: https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
    """
    if type(X) in [pd.DataFrame, pd.Series]:
        X = X.to_numpy()
    if type(y) in [pd.DataFrame, pd.Series]:
        y = y.to_numpy()
    fig = plt.figure(figsize=figsize)
    X1_range = X[:, 0].max() - X[:, 0].min()
    X2_range = X[:, 1].max() - X[:, 1].min()
    boundary_pct = 0.10
    X1, X2 = np.meshgrid(np.linspace(start=(X[:, 0].min() - boundary_pct*X1_range), stop=(X[:, 0].max() + boundary_pct*X1_range), num=500),
                         np.linspace(start=(X[:, 1].min() - boundary_pct*X2_range), stop=(X[:, 1].max() + boundary_pct*X2_range), num=500))
    colors = ('red', 'green')
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    plt.contourf(X1, X2, classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.1, cmap=cmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for counter, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0],
                    X[y == j, 1],
                    alpha=0.9,
                    c=colors[counter],
                    label=y_classes[counter])
    plt.title(title)
    plt.xlabel(X1_lab)
    plt.ylabel(X2_lab)
    plt.legend()
    fig.tight_layout()
    plt.show()


def Matthew_Correlation_Coefficient(TP, TN, FP, FN):
    """
    A correlation coefficient between the observed and predicted classifications
    Least influenced by imbalanced data.
    Range: -1 ~ 1
    1 = perfect prediction
    0 = andom prediction
    -1 = worst possible prediction.
    """
    num = (TP*TN)-(FP*FN)
    denom = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
    if denom != 0:
        MCC = num / denom
    else:
        MCC = None
    return MCC

def plot_confusion_matrix(y_true,
                          y_pred,
                          y_classes='auto',
                          figsize=(9, 10)):
    """

    This function plots the confusion matrix, along with key statistics, and returns accuracy.

    Required arguments:
        - y_true:     An array of shape (m_sample,); the labels could be {0,1}
        - y_pred:     An array of shape (m_sample,); the labels could be {0,1}

    Optional arguments:
        - y_classes:  A list, the y_classes to be displayed. if auto and two classes, it would be ('y=0', 'y=1') and so on
        - figsize:    A tuple, the figure size. reference: plt.rcParams.get('figure.figsize')

    """

    if y_classes == 'auto':
        y_classes = [f"y={y}" for y in np.unique(y_true)]

    cm = confusion_matrix(y_true, y_pred)

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
        if (TP+FP) !=0:
            precision = TP / (TP+FP)
            precision_text = f"{precision:.3f}"
        else:
            precision = None
            precision_text = "nan"
        sensitivity = TPR = recall = TP / (TP+FN)
        # or, f1_score = 2*precision*recall / (precision + recall) # f1_score is sensitive to class imbalance
        f1_score = TP / (TP + 0.5*(FP+FN))
        FPR = FP / (TN+FP)
        specificity = 1 - FPR
        geometric_mean = (TPR * (1-FPR)) ** 0.5 # GM measures the balance between classification performances on both the majority and minority classes. Insensitive to imbalance classes
        MCC = Matthew_Correlation_Coefficient(TP, TN, FP, FN)
        if MCC is None:
            MCC_text = 'nan'
        else:
            MCC_text = f"{MCC:.3f}"
        stats_text = f"\n\nAccuracy(higher TP and TN) = (TP+TN)/Total = {accuracy:0.3f}\nF1 Score(lower FP and FN) = TP/(TP+0.5*(FP+FN)) = {f1_score:0.3f}\nG-Mean = sqrt(TPR*(1-FPR)) = {geometric_mean:0.3f}\nMatthew's Correlation Coefficient = {MCC_text}\n\nTPR/recall/sensitivity = 1-FNR = p($y_{{pred}}$=1 | $y_{{true}}$=1) = {recall:0.3f}\nFPR = p($y_{{pred}}$=1 | $y_{{true}}$=0) = {FPR:0.3f}\n\nPrecision = 1-FDR = p($y_{{true}}$=1 | $y_{{pred}}$=1) = {precision_text}"
    else:
        stats_text = f"\n\nAccuracy={accuracy:0.3f}"

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
    plt.title('Confusion Matrix')

    fig.tight_layout()
    plt.show()

    plt.rcParams.update({'font.size': old_font_size})

    print("\n", classification_report(y_true, y_pred, target_names=y_classes))

    return accuracy,


def plot_ROC_curve(y_true,
                   y_pred_score,
                   y_pos_label=1,
                   figsize=(8, 7),
                   model_name='Binary Classifier',
                   plot_threshold=True):
    """

    This function plots the ROC (Receiver operating characteristic) curve, along with statistics.

    Required arguments:
        - y_true:       An array of shape (m_sample,); the labels could be {0,1}
        - y_pred_score: An array of shape (m_sample,); the probability estimates of the positive class

    Optional arguments:
        - y_pos_label:    The label of the positive class (default = 1)
        - figsize:        A tuple, the figure size. reference: plt.rcParams.get('figure.figsize')
        - model_name:     A string
        - plot_threshold: A boolean, whether to plot threshold or not

    """

    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_score, pos_label=y_pos_label)
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

    if plot_threshold:
        for threshold in (0.01, 0.50, 0.99):
            cm = confusion_matrix((y_true == y_pos_label).astype(bool),
                                  (y_pred_score >= threshold).astype(bool))
            TP = cm[1, 1]
            TN = cm[0, 0]
            FP = cm[0, 1]
            FN = cm[1, 0]
            TPR = TP / (TP+FN)
            FPR = FP / (TN+FP)
            plt.plot([FPR], [TPR], marker='x', markersize=10,
                     color="red", label=f"d={threshold:4.2f}")
            plt.annotate(text=f"d={threshold:4.2f}", xy=(
                FPR+0.01, TPR+0.01), color="red")

    fig.tight_layout()
    plt.show()

    plt.rcParams.update({'font.size': old_font_size})


def plot_PR_curve(fitted_model,
                  X,
                  y_true,
                  y_pred_score,
                  y_pos_label=1,
                  figsize=(8, 7),
                  model_name='Binary Classifier',
                  plot_threshold=True):
    """

    This function plots the precision-recall curve, along with statistics.

    Required arguments:
        - fitted_model: A fitted classifier instance
        - X:            A matrix of m_samples x n_features
        - y_true:       An array of shape (m_sample,); the labels could be {0,1}
        - y_pred_score: An array of shape (m_sample,); the probability estimates of the positive class

    Optional arguments:
        - y_pos_label:    The label of the positive class (default = 1)
        - figsize:        A tuple, the figure size. reference: plt.rcParams.get('figure.figsize')
        - model_name:     A string
        - plot_threshold: A boolean, whether to plot threshold or not

    """

    # AP is the area under the PR curve
    AP = average_precision_score(
        y_true=y_true, y_score=y_pred_score, pos_label=y_pos_label, average='macro')

    old_font_size = plt.rcParams.get('font.size')
    plt.rcParams.update({'font.size': __font_size__})

    fig, ax = plt.subplots(figsize=figsize)

    plot_precision_recall_curve(estimator=fitted_model, X=X, y=y_true,
                                response_method='predict_proba', ax=ax, label=f"{model_name} (AP = {AP:0.2f})")
    ax.set_xlabel('Recall = p($y_{pred}$=1 | $y_{true}$=1)')
    ax.set_ylabel('Precision = p($y_{true}$=1 | $y_{pred}$=1)')
    ax.set_title('Precision-Recall Curve')

    if plot_threshold:
        for threshold in (0.01, 0.50, 0.99):
            cm = confusion_matrix((y_true == y_pos_label).astype(bool), (y_pred_score >= threshold).astype(bool))
            TP = cm[1, 1]
            TN = cm[0, 0]
            FP = cm[0, 1]
            FN = cm[1, 0]
            TPR = TP / (TP+FN)
            FPR = FP / (TN+FP)
            if TP+FP == 0:
                precision = None
            else:
                precision = TP / (TP+FP)
            recall = TP / (TP+FN)
            if precision is not None:
                plt.plot([recall], [precision], marker='x', markersize=10, color="red", label=f"d={threshold:4.2f}")
                plt.annotate(text=f"d={threshold:4.2f}", xy=(recall+0.01, precision+0.01), color="red")

    fig.tight_layout()
    plt.show()

    plt.rcParams.update({'font.size': old_font_size})


def plot_ROC_and_PR_curves(fitted_model,
                           X,
                           y_true,
                           y_pred_score,
                           y_pos_label=1,
                           figsize=(8, 7),
                           model_name='Binary Classifier',
                           plot_threshold=True):
    """

    This function plots both the ROC and the precision-recall curves, along with statistics.

    Required arguments:
        - fitted_model: A fitted classifier instance
        - X:            A matrix of m_samples x n_features
        - y_true:       An array of shape (m_sample,); the labels could be {0,1}
        - y_pred_score: An array of shape (m_sample,); the probability estimates of the positive class

    Optional arguments:
        - y_pos_label:    The label of the positive class (default = 1)
        - figsize:        A tuple, the figure size. reference: plt.rcParams.get('figure.figsize')
        - model_name:     A string
        - plot_threshold: A boolean, whether to plot threshold or not

    """
    plot_ROC_curve(y_true=y_true, y_pred_score=y_pred_score,
                   y_pos_label=y_pos_label, figsize=figsize, model_name=model_name, plot_threshold=plot_threshold)

    plot_PR_curve(fitted_model=fitted_model, X=X,
                  y_true=y_true, y_pred_score=y_pred_score,
                  y_pos_label=y_pos_label, figsize=figsize, model_name=model_name, plot_threshold=plot_threshold)


def demo():
    """

    This function provides a demo of the major functions in this module.

    Required arguments:
        None

    """
    print(
        f"------------------------------------------------\n"
        f"This demo uses a made-up classification dataset.\n")
    from sklearn import datasets
    X, y = datasets.make_classification(
        n_samples=5000, n_features=30, n_classes=2, class_sep=0.8, random_state=1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    from ..naive_bayes import naive_bayes_Gaussian
    model = naive_bayes_Gaussian().fit(X_train, y_train)
    y_pred_score = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    accuracy = plot_confusion_matrix(y_true=y_test, y_pred=y_pred)
    plot_ROC_and_PR_curves(fitted_model=model, X=X_test, y_true=y_test, y_pred_score=y_pred_score[:, 1], model_name='Gaussian NB', plot_threshold=True)

