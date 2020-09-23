# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


def kNN_classifier(*args, **kwargs):
    """
    """
    return KNeighborsClassifier(*args, **kwargs)


def visualize_kNN_classifier(classifier, X_set, y_set, figsize=(8, 7)):
    """
    # reference: https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
    """
    fig = plt.figure(figsize=figsize)
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max(
    ) + 1, step=0.01), np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
        X1.shape), alpha=0.5, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], alpha=0.5,
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('K-Nearest Neighbors')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    fig.tight_layout()
    plt.show()


def demo():
    from ..datasets import public_dataset
    data = public_dataset(name='Social_Network_Ads')
    X = data[['Age', 'EstimatedSalary']].to_numpy()
    y = data['Purchased'].to_numpy()

    from sklearn.model_selection import train_test_split, GridSearchCV
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()  # removing the mean and scaling to unit variance
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # create pipeline
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps=[('scaler',
                                StandardScaler(with_mean=True, with_std=True)),
                               ('classifier',
                                kNN_classifier(n_neighbors=5, weights='uniform', p=2, metric='minkowski')),
                               ])

    # pipeline parameters to tune
    hyperparameters = {
        'scaler__with_mean': [True],
        'scaler__with_std': [True],
        'classifier__n_neighbors': range(5, 20),
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [2**0, 2**0.25, 2**0.5, 2**0.75, 2**1, 2**1.25, 2**1.5, 2**1.75, 2**2, 2**2.25, 2**2.5, 2**2.75, 2**3],
        'classifier__metric': ['minkowski'],
    }
    grid = GridSearchCV(
        pipeline,
        hyperparameters,  # parameters to tune via cross validation
        refit=True,       # fit using all data, on the best detected classifier
        n_jobs=-1,
        scoring='accuracy',
        cv=5,
    )
    classifier_grid = grid.fit(X_train, y_train)
    print(
        f"Using a grid search and a kNN classifier, the best hyperparameters were found as following:\n"
        f"Step1: scaler: StandardScaler(with_mean={repr(classifier_grid.best_params_['scaler__with_mean'])}, with_std={repr(classifier_grid.best_params_['scaler__with_std'])});\n"
        f"Step2: classifier: kNN_classifier(n_neighbors={repr(classifier_grid.best_params_['classifier__n_neighbors'])}, weights={repr(classifier_grid.best_params_['classifier__weights'])}, p={repr(classifier_grid.best_params_['classifier__p'])}, metric={repr(classifier_grid.best_params_['classifier__metric'])}).\n")

    y_pred = classifier_grid.best_estimator_.predict(X_test)
    y_pred_score = classifier_grid.best_estimator_.predict_proba(X_test)

    from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, y_classes=(
        'not_purchased (y=0)', 'purchased (y=1)'))
    plot_ROC_and_PR_curves(fitted_model=classifier_grid, X=X_test,
                           y_true=y_test, y_pred_score=y_pred_score[:, 1], y_pos_label=1, model_name='kNN')

    visualize_kNN_classifier(classifier_grid.best_estimator_, X_train, y_train)
    visualize_kNN_classifier(classifier_grid.best_estimator_, X_test, y_test)

