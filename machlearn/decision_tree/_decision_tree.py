# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def decision_tree(*args, **kwargs):
    return DecisionTreeClassifier(*args, **kwargs)

def bagging(*args, **kwargs):
    return BaggingClassifier(*args, **kwargs)


def GBM(*args, **kwargs):
    """
    The idea of boosting: one is weak, together is strong, iterative training leads to the best model.
    Strategy: To improve the predictive power by training a sequence of weak models. Each additional weak model is to compensate the weaknesses of its predecessors.

    GBM: Gradient Boosting Machines, including XGBOOST
    """
    return GradientBoostingClassifier(*args, **kwargs)


def AdaBoost(*args, **kwargs):
    """
    The idea of boosting: one is weak, together is strong, iterative training leads to the best model.
    Strategy: To improve the predictive power by training a sequence of weak models. Each additional weak model is to compensate the weaknesses of its predecessors.

    AdaBoost, Adaptive Boosting: at every step the sample distribution was adapted to put more weight on misclassified samples and less weight on correctly classified samples.
    """
    return AdaBoostClassifier(*args, **kwargs)


def _DT_demo_Social_Network_Ads(classifier_func="decision_tree"): # DT: decision_tree
    """
    classifier_func: "decision_tree", "GBM"
    """
    from ..datasets import public_dataset
    data = public_dataset(name='Social_Network_Ads')
    X = data[['Age', 'EstimatedSalary']].to_numpy()
    y = data['Purchased'].to_numpy()
    y_classes = ['not_purchased (y=0)', 'purchased (y=1)']

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
    if classifier_func == "decision_tree":
        pipeline = Pipeline(steps=[('scaler', StandardScaler(with_mean=True, with_std=True)),
                                   ('classifier', decision_tree(max_depth=1, random_state=123)),  # default criterion = 'gini'
                                   ])

        # pipeline parameters to tune
        hyperparameters={
            'scaler__with_mean': [True],
            'scaler__with_std': [True],
            'classifier__criterion': ("gini", "entropy"),
            'classifier__max_depth': range(1, 10),
        }

        model_name = "Decision Tree"
    
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
    criterion=classifier_grid.best_params_['classifier__criterion']
    max_depth=classifier_grid.best_params_['classifier__max_depth']
    print(
        f"Using a grid search and a {model_name} classifier, the best hyperparameters were found as following:\n"
        f"Step1: scaler: StandardScaler(with_mean={repr(classifier_grid.best_params_['scaler__with_mean'])}, with_std={repr(classifier_grid.best_params_['scaler__with_std'])});\n"
        f"Step2: classifier: {classifier_func}(criterion={repr(criterion)}, max_depth={repr(max_depth)}).\n")
    ########################################################################################################################

    y_pred=classifier_grid.predict(X_test)
    y_pred_score=classifier_grid.predict_proba(X_test)

    # Model evaluation
    from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves, visualize_classifier_decision_boundary_with_two_features
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, y_classes=y_classes)
    plot_ROC_and_PR_curves(fitted_model=classifier_grid, X=X_test,
                           y_true=y_test, y_pred_score=y_pred_score[:, 1], y_pos_label=1, model_name=f"{model_name}")

    visualize_classifier_decision_boundary_with_two_features(
        classifier_grid, X_train, y_train, y_classes, title=f"{model_name} / training set", X1_lab='Age', X2_lab='Estimated Salary')
    visualize_classifier_decision_boundary_with_two_features(
        classifier_grid, X_test,  y_test,  y_classes, title=f"{model_name} / testing set",  X1_lab='Age', X2_lab='Estimated Salary')

    # Plotting the tree
    if classifier_func == "decision_tree":
        from sklearn.tree import export_graphviz
        import io
        import pydotplus
        # from IPython.display import Image
        dot_data=io.StringIO()
        feature_cols=['Age', 'EstimatedSalary']
        export_graphviz(classifier_grid.best_estimator_.steps[1][1], out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
        graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
        image=graph.create_png()
        from PIL import Image
        Image.open(io.BytesIO(image)).show()


def demo(dataset="Social_Network_Ads", classifier_func="decision_tree"):
    """
    This function provides a demo of selected functions in this module.

    Required arguments:
        - dataset:         A string. Possible values: "Social_Network_Ads"
        - classifier_func: A string. Possible values: "decision_tree", "GBM"
    """
    if dataset == "Social_Network_Ads":
        _DT_demo_Social_Network_Ads(classifier_func=classifier_func)
    else:
        raise TypeError(f"dataset [{dataset}] is not defined")
