# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from machlearn import decision_tree as DT

DT.demo(dataset = "iris", classifier_func = "decision_tree")
DT.demo(dataset = "bank_note_authentication", classifier_func = "decision_tree")

DT.demo(dataset = "Social_Network_Ads", classifier_func = "decision_tree")
DT.demo(dataset = "Social_Network_Ads", classifier_func = "GBM")
DT.demo(dataset = "Social_Network_Ads", classifier_func = "AdaBoost")
DT.demo(dataset = "Social_Network_Ads", classifier_func = "bagging")

