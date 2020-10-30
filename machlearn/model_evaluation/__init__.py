# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from ._model_evaluation import test_for_multicollinearity, evaluate_continuous_prediction, visualize_classifier_decision_boundary_with_two_features, plot_confusion_matrix, plot_ROC_curve, plot_PR_curve, plot_ROC_and_PR_curves, demo, demo_CV

# this is for "from <package_name>.model_evaluation import *"
__all__ = ["test_for_multicollinearity",
           "evaluate_continuous_prediction",
           "visualize_classifier_decision_boundary_with_two_features",
           "plot_confusion_matrix",
           "plot_ROC_curve",
           "plot_PR_curve",
           "plot_ROC_and_PR_curves",
           "demo",
           "demo_CV"]
