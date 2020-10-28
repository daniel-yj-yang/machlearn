# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause


def LinearReg_statsmodels(y, X):
    """
    - Required arguments:
        y, X: pandas series
    """
    X_original = X

    import statsmodels.api as sm
    X = sm.add_constant(X) # fit_intercept
    model = sm.OLS(y, X).fit()
    print(f"Unstandardized estimates:\n{model.summary()}\n")
    #predictions = model.predict(X)

    from sklearn.preprocessing import scale
    X_scaled = scale(X_original)
    y_scaled = scale(y)
    model_scaled = sm.OLS(y_scaled, X_scaled).fit()
    print(f"Standardized estimates:\n{model_scaled.summary()}\n")

def LinearReg_normal_equation(y, X):
    """
    - Required arguments:
        y, X: pandas series
    """
    X_original = X

    import statsmodels.api as sm
    X = sm.add_constant(X)
    from numpy.linalg import inv
    theta = inv(X.T.dot(X)).dot(X.T).dot(y)
    print(f"Unstandardized estimates:\n{theta}")

    from sklearn.preprocessing import scale
    X_scaled = scale(X_original)
    y_scaled = scale(y)
    theta_scaled = inv(X_scaled.T.dot(X_scaled)).dot(X_scaled.T).dot(y_scaled)
    print(f"Standardized estimates:\n{theta_scaled}")

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

    print("*** Solutions using statsmodel ***\n")
    LinearReg_statsmodels(y, X)

    print("*** Solutions using normal equation ***\n")
    LinearReg_normal_equation(y, X)



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
