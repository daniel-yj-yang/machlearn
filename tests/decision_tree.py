# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from machlearn import decision_tree as DT

DT.demo_DT_from_scratch()

DT.demo_metrics()

DT.demo(dataset = "iris", classifier_func = "decision_tree")
DT.demo(dataset = "bank_note_authentication", classifier_func = "decision_tree")

DT.demo(dataset = "Social_Network_Ads", classifier_func = "decision_tree")
DT.demo(dataset = "Social_Network_Ads", classifier_func = "random_forest")
DT.demo(dataset = "Social_Network_Ads", classifier_func = "bagging")
DT.demo(dataset = "Social_Network_Ads", classifier_func = "AdaBoost")
DT.demo(dataset = "Social_Network_Ads", classifier_func = "GBM")
