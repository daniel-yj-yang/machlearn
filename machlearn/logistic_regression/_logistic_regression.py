# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause



def LR_statsmodels(y_pd_series, X_pd_series):
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
    print(result.summary2())
    return (result.params.values)


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
        y = data['Purchased']
        y_classes = ['not_purchased (y=0)', 'purchased (y=1)']

    LR_statsmodels(y, X)



def demo(dataset="Social_Network_Ads"):
    """
    This function provides a demo of selected functions in this module.

    Required argument:
        - dataset:         A string. Possible values: "Social_Network_Ads"
    """

    available_datasets = ("Social_Network_Ads",)

    if dataset in available_datasets:
        _demo(dataset = dataset)
    else:
        raise TypeError(f"dataset [{dataset}] is not defined")
