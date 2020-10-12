# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

def rnn():
    """
    Recurrent neural network
    """
    pass


def _neural_network_demo_Fashion_MNIST():
    from ..datasets import public_dataset
    X_train, y_train, X_test, y_test = public_dataset('Fashion_MNIST')


def demo(dataset="Fashion_MNIST"):
    """
    This function provides a demo of selected functions in this module.

    Required arguments:
        dataset: A string. Possible values: "Fashion_MNIST"

    """
    if dataset == "Fashion_MNIST":
        _neural_network_demo_Fashion_MNIST()
    else:
        raise TypeError(f"dataset [{dataset}] is not defined")
