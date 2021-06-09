# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause


def demo(dataset="Social_Network_Ads"):
    """

    This function provides a demo of selected functions in this module.

    Required arguments:
        dataset: A string. Possible values: "Social_Network_Ads", "iris"

    """
    if dataset == "Social_Network_Ads":
        _network_analysis_demo_Social_Network_Ads()
    elif dataset == "iris":
        _network_analysis_demo_iris()
    else:
        raise TypeError(f"dataset [{dataset}] is not defined")

