# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause



def LogisticReg_statsmodels(y_pd_series, X_pd_series):
    """
    - Required arguments:
        y_pd_series, X_pd_series
    """
    from statsmodels.discrete.discrete_model import Logit
    from statsmodels.tools import add_constant
    # fit_intercept
    model = Logit(endog=y_pd_series, exog=add_constant(X_pd_series))
    result = model.fit()
    print(result.summary())
    #print(result.summary2())
    return result.params.values


def _demo(dataset="Social_Network_Ads"):
    """
    """
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
        X = data[['Male', 'Age', 'EstimatedSalary']]
        #X = data[['Age']]
        y = data['Purchased']
        y_classes = ['not_purchased (y=0)', 'purchased (y=1)']

        params_values = LogisticReg_statsmodels(y, X)
        beta0 = params_values[0] # exp(beta0) = the baseline "odds_of_prob(y)" when X's=0:
        beta2 = params_values[2] # Age
        import math
        print(f"\nbeta0 (intercept) = {beta0:.2f}. Interpretation: exp^beta0 = {math.exp(beta0):.2f} is the baseline odds of prob(y) when X's = 0.")
        print(f"\nbeta2 (Age) = {beta2:.2f}. Interpretation: exp^beta2 = {math.exp(beta2):.2f} is the odds ratio associated with Age, meaning that for every 1-unit increase in Age, the odds of prob(y) will be OR = {math.exp(beta2):.2f} times bigger.\n")
        # beta > 0, meaning increase
        # beta = 0, meaning no change associated with this X
        # beta < 0, meaning decrease


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
        raise TypeError(f"dataset [{dataset}] is not defined")
