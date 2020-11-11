# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
import torch.nn as nn
import numpy as np

class linear_regression_torch(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.model = nn.Linear(in_size, out_size)
    def forward(self, x):
        return self.model(x)


def linear_regression_assumption_test(X, y, feature_names=None):
    """
    1. linearity in the relationship between X and y
    2. I.I.D. in residuals: residuals are Independently, Identically Distributed as normal
    3. for multiple linear regression, little or no multicollinearity

    References:
    - https: // jeffmacaluso.github.io/post/LinearRegressionAssumptions/
    - https: // towardsdatascience.com/assumptions-of-linear-regression-algorithm-ed9ea32224e1
    """

    model = linear_regression_sklearn().fit(X, y)
    y_pred = model.predict(X)
    data = pd.DataFrame({'y_true': y, 'y_pred': y_pred})
    residuals = y - y_pred

    from sklearn.preprocessing import scale
    X_scaled, y_scaled = scale(X), scale(y)
    model_scaled = linear_regression_sklearn().fit(X_scaled, y_scaled)
    y_pred_scaled = model_scaled.predict(X_scaled)
    data_scaled = pd.DataFrame({'y_true_scaled': y_scaled, 'y_pred_scaled': y_pred_scaled})
    residuals_scaled = y_scaled - y_pred_scaled

    print("------------------------------------------------------------------------------------------------")
    print("Assumption 1: Linearity in the relationship between X and y")
    print("to test this, make a scatter plot of y_pred vs. y_true, and check for linear relationship")
    import seaborn as sns
    sns.lmplot(x='y_true_scaled', y='y_pred_scaled', data=data_scaled, fit_reg=False, height=6)
    import matplotlib.pyplot as plt
    diagnoal_line_coords = np.linspace(min(min(y_scaled), min(y_pred_scaled)), max(max(y_scaled), max(y_pred_scaled)), 2)
    plt.plot(diagnoal_line_coords, diagnoal_line_coords, color='darkorange', linestyle='--')
    plt.title('Assumption 1: Linearity in the relationship between X and y')
    plt.suptitle('The dots should be scattered around the diagonal')
    plt.xlabel('Standardized y_true')
    plt.ylabel('Standardized y_pred')
    plt.tight_layout()
    plt.show()

    print("------------------------------------------------------------------------------------------------")
    print("Assumption 2: Residuals should be independently distributed")
    from statsmodels.stats.stattools import durbin_watson
    DW_stat = durbin_watson(residuals_scaled) # 0-4: range, 0-2: positive autocorrelation, 2-4: negative autocorrelation, 1.5-2.5: normal
    if DW_stat < 1.5:
        print(f"Durbin-Watson stat = {DW_stat:.2f} < 1.5, indicating positive autocorrelation present. Assumption was not met.")
    elif DW_stat > 2.5:
        print(f"Durbin-Watson stat = {DW_stat:.2f} > 2.5, indicating negative autocorrelation present. Assumption was not met.")
    else:
        print(f"Durbin-Watson stat = {DW_stat:.2f} is between 1.5 and 2.5, indicating little autocorrelation present. Assumption was met.")

    print("------------------------------------------------------------------------------------------------")
    print("Assumption 3: Residuals should be identically distributed for all predicted DV scores (homoscedasticity)")
    plt.subplots(figsize=(8, 6))
    ax = plt.subplot(111)
    plt.scatter(x=y_pred_scaled, y=residuals_scaled, alpha=1.0)
    ref_line_coords = np.linspace(y_pred_scaled.min(), y_pred_scaled.max(), 2)
    plt.plot(ref_line_coords, np.repeat(0, ref_line_coords.shape[0]), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Standardized y_pred')
    plt.ylabel('Standardized Residuals')
    plt.title('Assumption 3: Homoscedasticity')
    plt.suptitle('Residuals should vary randomly around 0 and have a constant variance for all values of y_pred')
    plt.show()

    print("------------------------------------------------------------------------------------------------")
    print("Assumption 4: Residuals should be normally distributed")
    from statsmodels.stats.diagnostic import normal_ad  # the Anderson-Darling test
    p_value = normal_ad(residuals_scaled)[1]
    if p_value < 0.05:
        print(f"p-value for the Anderson-Darling test was {p_value:.4f} < 0.05; residuals are not normally distributed. Assumption was not met.")
    else:
        print(f"p-value for the Anderson-Darling test was {p_value:.4f} >= 0.05, residuals are normally distributed. Assumption was met.")
    plt.subplots(figsize=(8, 6))
    plt.title('Assumption 4: Residuals should be normally distributed')
    sns.histplot(data=residuals_scaled, stat="density")
    sns.kdeplot(data=residuals_scaled, color="darkorange")
    plt.xlabel('Standardized Residuals')
    plt.show()

    print("------------------------------------------------------------------------------------------------")
    print("Assumption 5: For multiple linear regression, little or no multicollinearity")
    # If violated, issues with interpretability of the coefficients and the standard errors of the coefficients.
    from ..model_evaluation import test_for_multicollinearity
    test_for_multicollinearity(X, feature_names=feature_names)
    print("------------------------------------------------------------------------------------------------")


class OLS(object):
    def __init__(self, print_summary=True, use_statsmodels=False, alpha = 1000):
        super().__init__()
        self.y = None
        self.X = None
        self.print_summary = print_summary
        self.use_statsmodels = use_statsmodels
        self.alpha = alpha
        self.max_iter = 200000

    def model(self, y, X):
        pass
        
    def unstandardized_estimate(self):
        X_with_intercept = sm.add_constant(self.X)  # fit_intercept
        fitted_model = self.model(self.y, X_with_intercept)
        if self.print_summary:
            if self.use_statsmodels:
                try:
                    print(f"Unstandardized estimates:\n{fitted_model.summary()}\n")
                except:
                    print(f"Unstandardized estimates:\n{fitted_model.params}\n")
            else:
                print(f"Unstandardized estimates:\n{fitted_model.coef_}\n")
            y_pred = fitted_model.predict(X_with_intercept)
            from ..model_evaluation import evaluate_continuous_prediction
            RMSE, R_squared = evaluate_continuous_prediction(self.y, y_pred)
            print(f"RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")
        return fitted_model

    def standardized_estimate(self):
        from sklearn.preprocessing import scale
        import pandas as pd
        X_scaled = pd.DataFrame(scale(self.X), columns=self.X.columns)
        y_scaled = pd.Series(scale(self.y), name=self.y.name)
        fitted_model_scaled = self.model(y_scaled, X_scaled)
        if self.print_summary:
            if self.use_statsmodels:
                try:
                    print(f"Standardized estimates:\n{fitted_model_scaled.summary()}\n")
                except:
                    print(f"Standardized estimates:\n{fitted_model_scaled.params}\n")
            else:
                print(f"Standardized estimates:\n{fitted_model_scaled.coef_}\n")
            y_scaled_pred = fitted_model_scaled.predict(X_scaled)
            from ..model_evaluation import evaluate_continuous_prediction
            RMSE, R_squared = evaluate_continuous_prediction(y_scaled, y_scaled_pred)
            print(f"RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")
        return fitted_model_scaled
    
    def run(self, y, X, standardized_estimate=False):
        """
        - Required arguments:
            y: pandas series (1-dim)
            X: pandas data frame
        """
        self.y = y
        self.X = X
        import pandas as pd
        if isinstance(self.y, pd.DataFrame):
            self.y = self.y.squeeze()  # convert to data series
        estimates = self.unstandardized_estimate()
        if standardized_estimate:
            self.standardized_estimate()
        return estimates


def linear_regression_sklearn(*args, **kwargs):
    return linear_model.LinearRegression(*args, **kwargs)


class linear_regression(OLS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def model(self, y, X):
        if self.use_statsmodels:
            return sm.OLS(y, X).fit()
        else:
            return linear_regression_sklearn(fit_intercept=False).fit(X, y)


class linear_regression_normal_equation(OLS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_statsmodels = False

    def model(self, y, X):
        return normal_equation().fit(X, y)


class normal_equation(object):
    def __init__(self):
        super().__init__()
        self.coef_ = None
    
    def fit(self, X, y):
        from numpy.linalg import inv
        self.coef_ = inv(X.T.dot(X)).dot(X.T).dot(y)
        return self

    def predict(self, X):
        return X.dot(self.coef_)


class ridge_regression(OLS):
    def __init__(self, alpha=1000, *args, **kwargs):
        super().__init__(alpha=alpha, *args, **kwargs)
        self.alpha = alpha
        print(f"alpha = [{alpha}]")

    def model(self, y, X):
        if self.use_statsmodels:
            return sm.OLS(y, X).fit_regularized(method='elastic_net', alpha=self.alpha, L1_wt=0, refit = False)
        else:
            return linear_model.Ridge(alpha=self.alpha, fit_intercept=False).fit(X, y)


class lasso_regression(OLS):
    def __init__(self, alpha = 1000, *args, **kwargs):
        super().__init__(alpha = alpha, *args, **kwargs)
        self.alpha = alpha
        print(f"alpha = [{alpha}]")

    def model(self, y, X):
        if self.use_statsmodels:
            return sm.OLS(y, X).fit_regularized(method='elastic_net', alpha=self.alpha, L1_wt=1, maxiter=self.max_iter, refit = False)
        else:
            return linear_model.Lasso(alpha=self.alpha, max_iter=self.max_iter, fit_intercept=False).fit(X, y)


def identify_best_alpha_for_ridge_regression(y, X, alphas = [0.1, 1.0, 10.0]):
    from sklearn.linear_model import RidgeCV
    ridge_regression_cv = RidgeCV(alphas=alphas, fit_intercept=False)
    model_cv = ridge_regression_cv.fit(X, y)
    return model_cv.alpha_


def identify_best_alpha_for_lasso_regression(y, X, alphas=[0.1, 1.0, 10.0]):
    from sklearn.linear_model import LassoCV
    lasso_regression_cv = LassoCV(alphas=alphas, fit_intercept=False, max_iter=200000)
    model_cv = lasso_regression_cv.fit(X, y)
    return model_cv.alpha_


def _demo_regularization(dataset="Hitters", use_statsmodels=False):
    """
    """
    print('\nRegularization is to handle overfitting, which means the model fitted with the training data will not well generalize to the testing data.')
    print('Several possibilities would cause overfitting, including (a) multicolinearity among predictors and (b) model being too complex and having trivial predictors.')
    print('When (a) there is multicolinearity among predictors, we use L2 Regularization, which adds "squared magnitude" of coefficient (squared L2 norm) as a penalty term to the cost function.')
    print('When (b) model is too complex and has trivial predictors, we use L1 Regularization, which adds "magnitude" of coefficient (L1 norm) as a penalty term to the cost function.')
    print('\nL2 regularization is also known as Ridge regression, while L1 regularization is also known as Lasso regression, and a combination of them is known as elastic net.')
    print('After regularization, we would expect to see better generalization, including reduced RMSE and improved R^2.')
    print('After L2 regularization, we would expect to see smaller variances among the coefficient estimates, that is, the estimates less likely changing rapidly and hence more robust.')
    print('After L1 regularization, we would expect to see a simpler model with many coefficient estimates = 0.')
    print('For either L2 or L1 regularization, there is also a parameter called alpha (or lambda), which governs the amount of regularization. It takes GridCV (e.g., RidgeCV or LassoCV) to identify the optimal number of alpha (or lambda).\n')

    import pandas as pd
    import patsy

    if dataset == "boston":
        from ..datasets import public_dataset
        data = public_dataset(name="boston")
        print(f"{data.head()}\n")
        formula = 'MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT - 1'

    if dataset == "Longley":
        # https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html
        from statsmodels.datasets.longley import load_pandas
        y = load_pandas().endog
        X = load_pandas().exog
        data = pd.concat([y, X], axis=1) # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
        print(f"{data.head()}\n")
        formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR - 1' # -1 means no intercept

    if dataset == "Hitters":
        from ..datasets import public_dataset
        data = public_dataset(name="Hitters")
        data = data.dropna()
        print(f"{data.head()}\n")

        data = data.drop(['League', 'NewLeague', 'Division'], axis=1)
        formula = 'Salary ~ AtBat + Hits + HmRun + Runs + RBI + Walks + Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks + PutOuts + Assists + Errors - 1'  # -1 means no intercept
        #formula = 'Salary ~ League + NewLeague + Division + AtBat + Hits + HmRun + Runs + RBI + Walks + Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks + PutOuts + Assists + Errors - 1' # -1 means no intercept
    
    y, X = patsy.dmatrices(formula, data)
    feature_names = X.design_info.column_names
    X = pd.DataFrame(X, columns=feature_names)

    import numpy as np
    best_ridge_regression_alpha = identify_best_alpha_for_ridge_regression(y.ravel(), sm.add_constant(X), alphas=np.exp(np.linspace(-7,15,100000)))
    print(f"best alpha for ridge regression: {best_ridge_regression_alpha:.2f}")

    best_lasso_regression_alpha = identify_best_alpha_for_lasso_regression(y.ravel(), sm.add_constant(X), alphas=np.exp(np.linspace(-7,15, 10000)))
    print(f"best alpha for lasso regression: {best_lasso_regression_alpha:.2f}")

    from ..model_evaluation import test_for_multicollinearity
    test_for_multicollinearity(X, feature_names=feature_names)

    train = data.sample(frac=0.50, random_state=123)
    test = data[~data.isin(train).iloc[:, 0]]
    print(f"data size: training set = {len(train)}, testing set = {len(test)}, total = {len(data)}.")

    y_train, X_train = patsy.dmatrices(formula, train)
    y_test, X_test = patsy.dmatrices(formula, test)

    X_train = pd.DataFrame(X_train, columns = X_train.design_info.column_names )
    y_train = pd.DataFrame(y_train, columns = y_train.design_info.column_names )

    X_test = pd.DataFrame(X_test, columns = X_test.design_info.column_names )
    y_test = pd.DataFrame(y_test, columns = y_test.design_info.column_names )

    for i, model in enumerate([linear_regression, ridge_regression, lasso_regression]):
        print(f"------------------------------------------------------------------------------")
        print(f"{repr(model)}\n")
        #fitted_model = model.run(y, X)

        if i == 0:
            alpha = None
        elif i == 1:
            alpha = best_ridge_regression_alpha
        elif i == 2:
            alpha = best_lasso_regression_alpha

        fitted_model = model(print_summary = False, use_statsmodels = use_statsmodels, alpha = alpha).run(y_train, X_train)
        y_test_pred = fitted_model.predict(sm.add_constant(X_test))
        
        from ..model_evaluation import evaluate_continuous_prediction
        RMSE, R_squared = evaluate_continuous_prediction(y_test.squeeze(), y_test_pred)
        print(f"model performance with the testing set: RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")
        if use_statsmodels:
            print(fitted_model.params)
        else:
            print(fitted_model.coef_)
            variance = np.var(fitted_model.coef_)
            print(f"\nvariance among the coefficients: {variance:.2f}")


def _demo(dataset="marketing", use_statsmodels = False):
    """
    """
    from ..datasets import public_dataset

    if dataset == "marketing":
        data = public_dataset(name="marketing")
        print(f"{data.head()}\n")
        # pairplot
        import seaborn as sns
        sns.pairplot(data)
        import matplotlib.pyplot as plt
        plt.show()
        # X and y
        X = data[['youtube', 'facebook', 'newspaper']]
        y = data['sales']

    linear_regression_assumption_test(X, y, feature_names=['youtube', 'facebook', 'newspaper'])
    
    print("----------------------------------\n\n*** Solutions using python package ***\n")
    linear_regression(print_summary = True, use_statsmodels=use_statsmodels).run(y, X, standardized_estimate=True)

    print("---------------------------------------\n\n*** Solutions using normal equation ***\n")
    linear_regression_normal_equation(print_summary = True).run(y, X, standardized_estimate=True)


def demo_regularization(dataset="Hitters", use_statsmodels = False):
    """
    This function provides a demo of selected functions in this module.

    Required argument:
        - dataset:         A string. Possible values: "Longley", "Hitters", "boston"
        - use_statsmodels: boolean
    """
    available_datasets = ("Longley", "Hitters", "boston", )

    if dataset in available_datasets:
        return _demo_regularization(dataset = dataset, use_statsmodels = use_statsmodels)
    else:
        raise TypeError(f"dataset [{dataset}] is not defined")


def demo(dataset="marketing", use_statsmodels=False):
    """
    This function provides a demo of selected functions in this module.

    Required argument:
        - dataset:         A string. Possible values: "marketing"
    """

    available_datasets = ("marketing",)

    if dataset in available_datasets:
        _demo(dataset=dataset, use_statsmodels=use_statsmodels)
    else:
        raise TypeError(f"dataset [{dataset}] is not defined")


def demo_assumption_test():

    from ..datasets import public_dataset
    df = public_dataset(name="boston")
    print(f"{df.head()}\n")
    print("\nData1: boston")
    feature_names = df.columns.drop('MEDV')
    linear_regression_assumption_test(df[feature_names], df['MEDV'], feature_names=feature_names)

    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=df.shape[0], n_features=df.shape[1]-1, noise=100, random_state=10)
    print("\nData2: make-up data using make_regression()")
    linear_regression_assumption_test(X, y)

