# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause


from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import CCA


def principal_component_analysis(*args, **kwargs):
    return PCA(*args, **kwargs)


def independent_component_analysis(*args, **kwargs):
    return FastICA(*args, **kwargs)


def canonical_correlation_analysis(*args, **kwargs):
    return CCA(*args, **kwargs)


def demo():
    pass
