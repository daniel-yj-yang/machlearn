# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from sklearn.neural_network import MLPClassifier

def multi_layer_perceptron_classifier(*args, **kwargs):
    return MLPClassifier(*args, **kwargs)

def rnn():
    """
    Recurrent neural network
    """
    pass

def _demo(dataset="Social_Network_Ads"):
    from ..datasets import public_dataset
    if dataset == "Social_Network_Ads":
        data = public_dataset(name="Social_Network_Ads")
        print(f"{data.head()}\n")
        del data['User ID']
        # Recode the data: Gender as Male
        mapper = {'Male': 1, 'Female': 0}
        data['Male'] = data['Gender'].map(mapper)
        # pairplot
        import seaborn as sns
        sns.pairplot(data, hue="Purchased", markers=["o", "s"])
        import matplotlib.pyplot as plt
        plt.show()
        # X and y
        # dropping Male to simplify the analysis
        X = data[['Age', 'EstimatedSalary']]
        y = data['Purchased']
        y_classes = ['not_purchased (y=0)', 'purchased (y=1)']

    from sklearn.model_selection import train_test_split, GridSearchCV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    # Feature scaling
    from sklearn.preprocessing import StandardScaler

    # create pipeline
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(steps=[('scaler', StandardScaler(with_mean=True, with_std=True)),
                                ('classifier', multi_layer_perceptron_classifier(max_iter=300, random_state=123)),
                                ])

    # pipeline parameters to tune
    hyperparameters={
        'scaler__with_mean': [True],
        'scaler__with_std': [True],
        'classifier__alpha': (0.01,),
        'classifier__learning_rate_init': (0.001, 0.0001, 0.00001, 0.000001),
        'classifier__max_iter': (500, 1000, 2000, 3000),
    }

    model_name = "MLP Classifier"

    grid=GridSearchCV(
        pipeline,
        hyperparameters,  # parameters to tune via cross validation
        refit=True,       # fit using all data, on the best detected classifier
        n_jobs=-1,
        scoring='accuracy',
        cv=5,
    )

    classifier_grid=grid.fit(X_train, y_train)

    alpha = classifier_grid.best_params_['classifier__alpha']
    learning_rate_init = classifier_grid.best_params_['classifier__learning_rate_init']
    max_iter=classifier_grid.best_params_['classifier__max_iter']
    print(
        f"Using a grid search and a {model_name} classifier, the best hyperparameters were found as following:\n"
        f"Step1: scaler: StandardScaler(with_mean={repr(classifier_grid.best_params_['scaler__with_mean'])}, with_std={repr(classifier_grid.best_params_['scaler__with_std'])});\n"
        f"Step2: classifier: multi_layer_perceptron_classifier(alpha={alpha}, learning_rate_init={learning_rate_init}, max_iter={max_iter}).\n")

    model_desc = "Neural Network MLP Classifier"

    y_pred = classifier_grid.predict(X_test)
    y_pred_score = classifier_grid.predict_proba(X_test)

    # Model evaluation
    from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves, visualize_classifier_decision_boundary_with_two_features
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, y_classes=y_classes)

    if dataset in ['Social_Network_Ads', ]:
        plot_ROC_and_PR_curves(fitted_model=classifier_grid, X=X_test,
                               y_true=y_test, y_pred_score=y_pred_score[:, 1], y_pos_label=1, model_name=f"{model_name}")

    if dataset in ['Social_Network_Ads', ]:
        visualize_classifier_decision_boundary_with_two_features(
            classifier_grid, X_train.to_numpy(), y_train.to_numpy(), y_classes, title=f"{model_desc} / training set", X1_lab='Age', X2_lab='Estimated Salary')
        visualize_classifier_decision_boundary_with_two_features(
            classifier_grid, X_test.to_numpy(),  y_test.to_numpy(),  y_classes, title=f"{model_desc} / testing set",  X1_lab='Age', X2_lab='Estimated Salary')


def _neural_network_demo_Fashion_MNIST():
    from ..datasets import public_dataset
    X_train, y_train, X_test, y_test = public_dataset('Fashion_MNIST')


def demo(dataset="Social_Network_Ads"):
    """
    This function provides a demo of selected functions in this module.

    Required arguments:
        dataset: A string. Possible values: "Social_Network_Ads", "Fashion_MNIST"

    """
    if dataset == "Social_Network_Ads":
        _demo(dataset=dataset)
    elif dataset == "Fashion_MNIST":
        _neural_network_demo_Fashion_MNIST()
    else:
        raise TypeError(f"dataset [{dataset}] is not defined")
