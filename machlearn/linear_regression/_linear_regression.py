# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause


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
        # formula = 'Salary ~ League + NewLeague + Division + AtBat + Hits + HmRun + Runs + RBI + Walks + Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks + PutOuts + Assists + Errors - 1'
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

        train_model = LinearReg_statsmodels(y_train, X_train, print_output = False)

        import statsmodels.api as sm
        X_test = sm.add_constant(X_test)  # fit_intercept
        y_test_pred = train_model.predict(X_test)

        from ..model_evaluation import evaluate_continuous_prediction
        RMSE, R_squared = evaluate_continuous_prediction(y_test.squeeze(), y_test_pred)
        print(f"Before regularation, model performance with the testing set: RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")

        # to implement: Ridge, Lasso regression


def LinearReg_statsmodels(y, X, print_output = True):
    """
    - Required arguments:
        y: pandas series (1-dim)
        X: pandas data frame
    """
    import pandas as pd
    if isinstance(y, pd.DataFrame):
        y = y.squeeze() # convert to data series

    from ..model_evaluation import evaluate_continuous_prediction

    X_original = X

    import statsmodels.api as sm
    X = sm.add_constant(X) # fit_intercept
    model = sm.OLS(y, X).fit()
    if print_output:
        print(f"Unstandardized estimates:\n{model.summary()}\n")
        y_pred = model.predict(X)
        RMSE, R_squared = evaluate_continuous_prediction(y, y_pred)
        print(f"RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")

    from sklearn.preprocessing import scale
    X_scaled = scale(X_original)
    y_scaled = scale(y)
    X_scaled = pd.DataFrame(X_scaled, columns = X_original.columns )
    y_scaled = pd.Series(y_scaled, name = y.name )
    model_scaled = sm.OLS(y_scaled, X_scaled).fit()
    if print_output:
        print(f"Standardized estimates:\n{model_scaled.summary()}\n")
        y_scaled_pred = model_scaled.predict(X_scaled)
        RMSE, R_squared = evaluate_continuous_prediction(y_scaled, y_scaled_pred)
        print(f"RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")

    return model


def LinearReg_normal_equation(y, X):
    """
    - Required arguments:
        y: pandas series (1-dim)
        X: pandas data frame
    """
    import pandas as pd
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()  # convert to data series

    from ..model_evaluation import evaluate_continuous_prediction

    X_original = X

    import statsmodels.api as sm
    X = sm.add_constant(X)
    from numpy.linalg import inv
    theta = inv(X.T.dot(X)).dot(X.T).dot(y)
    print(f"Unstandardized estimates:\n{theta}")
    y_pred = X.dot(theta)
    RMSE, R_squared = evaluate_continuous_prediction(y, y_pred)
    print(f"RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")

    from sklearn.preprocessing import scale
    X_scaled = scale(X_original)
    y_scaled = scale(y)
    X_scaled = pd.DataFrame(X_scaled, columns = X_original.columns )
    y_scaled = pd.Series(y_scaled, name = y.name )
    theta_scaled = inv(X_scaled.T.dot(X_scaled)).dot(X_scaled.T).dot(y_scaled)
    print(f"Standardized estimates:\n{theta_scaled}")
    y_scaled_pred = X_scaled.dot(theta_scaled)
    RMSE, R_squared = evaluate_continuous_prediction(y_scaled, y_scaled_pred)
    print(f"RMSE = {RMSE:.3f}, R-squared = {R_squared:.3f}.\n")


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
    LinearReg_statsmodels(y, X)

    print("---------------------------------------\n\n*** Solutions using normal equation ***\n")
    LinearReg_normal_equation(y, X)


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
