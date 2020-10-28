# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import statsmodels.api as sm

class OLS():
    def __init__(self):
        self.y = None
        self.X = None
        self.print_output = None

    def model(self, y, X):
        pass
        
    def unstandardized_estimate(self):
        X_with_intercept = sm.add_constant(self.X)  # fit_intercept
        model = self.model(self.y, X_with_intercept)
        if self.print_output:
            print(f"Unstandardized estimates:\n{model.summary()}\n")
            y_pred = model.predict(X_with_intercept)
            from ..model_evaluation import evaluate_continuous_prediction
            RMSE, R_squared = evaluate_continuous_prediction(self.y, y_pred)
            print(f"RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")
        return model

    def standardized_estimate(self):
        from sklearn.preprocessing import scale
        import pandas as pd
        X_scaled = scale(self.X)
        y_scaled = scale(self.y)
        X_scaled = pd.DataFrame(X_scaled, columns=self.X.columns)
        y_scaled = pd.Series(y_scaled, name=self.y.name)
        model_scaled = self.model(y_scaled, X_scaled)
        if self.print_output:
            print(f"Standardized estimates:\n{model_scaled.summary()}\n")
            y_scaled_pred = model_scaled.predict(X_scaled)
            from ..model_evaluation import evaluate_continuous_prediction
            RMSE, R_squared = evaluate_continuous_prediction(y_scaled, y_scaled_pred)
            print(f"RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")
        return model_scaled
    
    def run(self, y, X, print_output=True):
        """
        - Required arguments:
            y: pandas series (1-dim)
            X: pandas data frame
        """
        self.y = y
        self.X = X
        self.print_output = print_output
        import pandas as pd
        if isinstance(self.y, pd.DataFrame):
            self.y = self.y.squeeze()  # convert to data series
        model = self.unstandardized_estimate()
        self.standardized_estimate()
        return model


class LinearReg_statsmodels(OLS):
    def __init__(self):
        super().__init__()
    def model(self, y, X):
        return sm.OLS(y, X).fit()


class Ridge_regression(OLS):
    def __init__(self):
        super().__init__()

    def model(self, y, X):
        return sm.OLS(y, X).fit_regularized(L1_wt=0)


class Lasso_regression(OLS):
    def __init__(self):
        super().__init__()

    def model(self, y, X):
        return sm.OLS(y, X).fit_regularized(L1_wt=1)


class LinearReg_normal_equation(OLS):
    def __init__(self):
        super().__init__()

    def model(self, y, X):
        from numpy.linalg import inv
        theta = inv(X.T.dot(X)).dot(X.T).dot(y)
        return theta

    def unstandardized_estimate(self):
        X_with_intercept = sm.add_constant(self.X)  # fit_intercept
        theta = self.model(self.y, X_with_intercept)
        if self.print_output:
            print(f"Unstandardized estimates:\n{theta}")
            y_pred = X_with_intercept.dot(theta)
            from ..model_evaluation import evaluate_continuous_prediction
            RMSE, R_squared = evaluate_continuous_prediction(self.y, y_pred)
            print(f"RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")
        return theta

    def standardized_estimate(self):
        from sklearn.preprocessing import scale
        import pandas as pd
        X_scaled = scale(self.X)
        y_scaled = scale(self.y)
        X_scaled = pd.DataFrame(X_scaled, columns=self.X.columns)
        y_scaled = pd.Series(y_scaled, name=self.y.name)
        theta_scaled = self.model(y_scaled, X_scaled)
        if self.print_output:
            print(f"Standardized estimates:\n{theta_scaled}")
            y_scaled_pred = X_scaled.dot(theta_scaled)
            from ..model_evaluation import evaluate_continuous_prediction
            RMSE, R_squared = evaluate_continuous_prediction(y_scaled, y_scaled_pred)
            print(f"RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")
        return theta_scaled


def _demo_regularization(dataset="Hitters"):
    """
    """
    from ..datasets import public_dataset

    if dataset == "Hitters":
        data = public_dataset(name="Hitters")
        data = data.dropna()
        print(f"{data.head()}\n")

        import patsy
        data = data.drop(['League', 'NewLeague', 'Division'], axis=1)
        #formula = 'Salary ~ League + NewLeague + Division + AtBat + Hits + HmRun + Runs + RBI + Walks + Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks + PutOuts + Assists + Errors - 1'
        formula = 'Salary ~ AtBat + Hits + HmRun + Runs + RBI + Walks + Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks + PutOuts + Assists + Errors - 1'
        y, X = patsy.dmatrices(formula, data)
        
        train = data.sample(frac=0.5, random_state=123)
        test = data[~data.isin(train).iloc[:, 0]]

        y_train, X_train = patsy.dmatrices(formula, train)
        y_test, X_test = patsy.dmatrices(formula, test)

        import pandas as pd
        X_train = pd.DataFrame(X_train, columns = X_train.design_info.column_names )
        y_train = pd.DataFrame(y_train, columns = y_train.design_info.column_names )

        X_test = pd.DataFrame(X_test, columns = X_test.design_info.column_names )
        y_test = pd.DataFrame(y_test, columns = y_test.design_info.column_names )

        for model in [LinearReg_statsmodels(), Ridge_regression(), Lasso_regression()]:
            trained_model = model.run(y_train, X_train, print_output = False)
            #print(dir(trained_model))
            y_test_pred = trained_model.predict(sm.add_constant(X_test)) # fit_intercept

            from ..model_evaluation import evaluate_continuous_prediction
            RMSE, R_squared = evaluate_continuous_prediction(y_test.squeeze(), y_test_pred)
            print(f"Model = {repr(model)}, model performance with the testing set: RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")



def _demo(dataset="marketing"):
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

    print("----------------------------------\n\n*** Solutions using statsmodel ***\n")
    LinearReg_statsmodels().run(y, X)

    print("---------------------------------------\n\n*** Solutions using normal equation ***\n")
    LinearReg_normal_equation().run(y, X)


def demo_regularization(dataset="Hitters"):
    """
    This function provides a demo of selected functions in this module.

    Required argument:
        - dataset:         A string. Possible values: "Hitters"
    """
    available_datasets = ("Hitters",)

    if dataset in available_datasets:
        return _demo_regularization(dataset = dataset)
    else:
        raise TypeError(f"dataset [{dataset}] is not defined")


def demo(dataset="marketing"):
    """
    This function provides a demo of selected functions in this module.

    Required argument:
        - dataset:         A string. Possible values: "marketing"
    """

    available_datasets = ("marketing",)

    if dataset in available_datasets:
        _demo(dataset = dataset)
    else:
        raise TypeError(f"dataset [{dataset}] is not defined")
