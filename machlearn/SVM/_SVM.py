# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from sklearn.svm import SVC


def SVM_classifier(*args, **kwargs):
    return SVC(*args, **kwargs)

#class SVM_classifier(SVC):
#    """
#    """
#    def __init__(self, *, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
#        super().__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, break_ties=break_ties, random_state=random_state)


def _SVM_demo_Social_Network_Ads():
    from ..datasets import public_dataset
    data = public_dataset(name='Social_Network_Ads')
    X = data[['Age', 'EstimatedSalary']].to_numpy()
    y = data['Purchased'].to_numpy()
    y_classes = ['not_purchased (y=0)', 'purchased (y=1)']

    from sklearn.model_selection import train_test_split, GridSearchCV
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123)

    from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()  # removing the mean and scaling to unit variance
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    # create pipeline
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps=[('scaler',
                                StandardScaler(with_mean=True, with_std=True)),
                               ('classifier',
                                SVM_classifier(probability=True)),
                               ])

    # pipeline parameters to tune
    hyperparameters = {
        'scaler__with_mean': [True],
        'scaler__with_std': [True],
        'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'classifier__gamma': ['scale', 'auto'],
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
    kernel = classifier_grid.best_params_['classifier__kernel']
    gamma = classifier_grid.best_params_['classifier__gamma']
    print(
        f"Using a grid search and a SVM classifier, the best hyperparameters were found as following:\n"
        f"Step1: scaler: StandardScaler(with_mean={repr(classifier_grid.best_params_['scaler__with_mean'])}, with_std={repr(classifier_grid.best_params_['scaler__with_std'])});\n"
        f"Step2: classifier: SVM_classifier(kernel={repr(kernel)}, gamma={repr(gamma)}).\n")

    y_pred = classifier_grid.predict(X_test)
    y_pred_score = classifier_grid.predict_proba(X_test)

    from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves, visualize_classifier_decision_boundary_with_two_features
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, y_classes=y_classes)
    plot_ROC_and_PR_curves(fitted_model=classifier_grid, X=X_test,
                           y_true=y_test, y_pred_score=y_pred_score[:, 1], y_pos_label=1, model_name=f"SVM kernel={kernel}")

    visualize_classifier_decision_boundary_with_two_features(
        classifier_grid, X_train, y_train, y_classes, title=f"SVM (kernel={kernel}) / training set", X1_lab='Age', X2_lab='Estimated Salary')
    visualize_classifier_decision_boundary_with_two_features(
        classifier_grid, X_test,  y_test,  y_classes, title=f"SVM (kernel={kernel}) / testing set",  X1_lab='Age', X2_lab='Estimated Salary')


def demo(dataset="Social_Network_Ads"):
    """

    This function provides a demo of selected functions in this module.

    Required arguments:
        dataset: A string. Possible values: "Social_Network_Ads"

    """
    if dataset == "Social_Network_Ads":
        _SVM_demo_Social_Network_Ads()
    else:
        raise TypeError(f"dataset [{dataset}] is not defined")
