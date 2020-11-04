# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from sklearn.linear_model import LogisticRegression

def logistic_regression_classifier(*args, **kwargs):
    return LogisticRegression(*args, **kwargs)


class logisticReg_statsmodels(object):
    def __init__(self):
        super().__init__()
    
    def run(self, y_pd_series, X_pd_DataFrame):
        """
        - Required arguments:
            y_pd_series, X_pd_DataFrame
        """
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        # fit_intercept
        model = Logit(endog=y_pd_series, exog=add_constant(X_pd_DataFrame))
        result = model.fit()
        print(result.summary())
        #print(result.summary2())
        return result.params.values


class logisticReg_sklearn(object):
    def __init__(self):
        super().__init__()
    
    def run(self, y_pd_series, X_pd_DataFrame):
        """
        - Required arguments:
            y_pd_series, X_pd_DataFrame
        """
        model = logistic_regression_classifier(solver='liblinear', fit_intercept=True, max_iter=1e5, tol=1e-8, C=1e10)
        model.fit(X_pd_DataFrame, y_pd_series)
        estimates = list(model.intercept_) + list(model.coef_[0]) # model.intercept_ is <class 'numpy.ndarray'>
        print(f"coefficient estimates: {['%.6f' % x for x in estimates]}")
        return estimates


def _demo(dataset="Social_Network_Ads"):
    """
    """
    from ..datasets import public_dataset

    if dataset == "Social_Network_Ads":
        data = public_dataset(name="Social_Network_Ads")
        print(f"{data.head()}\n")
        data = data.drop('User ID', 1)
        # Recode the data: Gender as Male
        mapper = {'Male': 1, 'Female': 0}
        data['Male'] = data['Gender'].map(mapper)
        # pairplot
        import seaborn as sns
        sns.pairplot(data, hue="Purchased", markers=["o", "s"])
        import matplotlib.pyplot as plt
        plt.show()
        # X and y
        #from sklearn.preprocessing import scale
        #X = scale(data[['Male', 'Age', 'EstimatedSalary']])
        X = data[['Male', 'Age', 'EstimatedSalary']]
        #X = data[['Age']]
        y = data['Purchased']
        y_classes = ['not_purchased (y=0)', 'purchased (y=1)']

        for model in [logisticReg_statsmodels, logisticReg_sklearn]:
            print(f"---------------------------------------------------------------------------------------------------------\nmodel: {repr(model)}.\n")
            params_values = model().run(y, X)
            beta0 = params_values[0] # exp(beta0) = the baseline "odds_of_prob(y)" when X's=0:
            beta2 = params_values[2] # Age
            import math
            print(f"\nbeta0 (intercept) = {beta0:.2f}. Interpretation: exp^beta0 = {math.exp(beta0):.2f} is the baseline odds of prob(y) when X's = 0.")
            print(f"\nbeta2 (Age) = {beta2:.2f}. Interpretation: exp^beta2 = {math.exp(beta2):.2f} is the odds ratio associated with Age, meaning that for every 1-unit increase in Age, the odds of prob(y) will be OR = {math.exp(beta2):.2f} times bigger.\n")
            # beta > 0, meaning increase
            # beta = 0, meaning no change associated with this X
            # beta < 0, meaning decrease

        print("---------------------------------------------------------------------------------------------------------")
        X = data[['Age', 'EstimatedSalary']].to_numpy()
        y = data['Purchased'].to_numpy()
        from sklearn.model_selection import train_test_split, GridSearchCV
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                   ('classifier', logistic_regression_classifier()), ])
        hyperparameters = {
            'scaler__with_mean': [True],
            'scaler__with_std': [True],
            'classifier__solver': ['liblinear'],
            'classifier__fit_intercept': [True],
            'classifier__max_iter': [1e5],
            'classifier__tol': [1e-8],
            'classifier__C': [1e10],
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

        y_pred = classifier_grid.predict(X_test)
        y_pred_score = classifier_grid.predict_proba(X_test)
            
        from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves, visualize_classifier_decision_boundary_with_two_features
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred, y_classes=y_classes)
        plot_ROC_and_PR_curves(fitted_model=classifier_grid, X=X_test, y_true=y_test, y_pred_score=y_pred_score[:, 1], y_pos_label=1, model_name=f"Logistic regression")

        visualize_classifier_decision_boundary_with_two_features(classifier_grid, X_train, y_train, y_classes, title=f"Logistic regression / training set", X1_lab='Age', X2_lab='Estimated Salary')
        visualize_classifier_decision_boundary_with_two_features(classifier_grid, X_test,  y_test,  y_classes, title=f"Logistic regression / testing set",  X1_lab='Age', X2_lab='Estimated Salary')



def demo(dataset="Social_Network_Ads"):
    """
    This function provides a demo of selected functions in this module.

    Required argument:
        - dataset:         A string. Possible values: "Social_Network_Ads"
    """

    available_datasets = ("Social_Network_Ads",)

    if dataset in available_datasets:
        return _demo(dataset = dataset)
    else:
        raise ValueError(f"dataset [{dataset}] is not defined")
