.. -*- mode: rst -*-

|BuildTest|_ |PythonVersion|_ |PyPi_version|_ |Downloads|_ |License|_

.. |BuildTest| image:: https://travis-ci.com/daniel-yj-yang/machlearn.svg?branch=master
.. _BuildTest: https://travis-ci.com/daniel-yj-yang/machlearn

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue

.. |PyPi_version| image:: https://img.shields.io/pypi/v/machlearn
.. _PyPi_version: https://pypi.python.org/pypi/machlearn

.. |Downloads| image:: https://pepy.tech/badge/machlearn
.. _Downloads: https://pepy.tech/project/machlearn

.. |License| image:: https://img.shields.io/pypi/l/machlearn
.. _License: https://pypi.python.org/pypi/machlearn


=====================================================
A Simple Yet Powerful Machine Learning Python Library
=====================================================

Install
-------

.. code-block:: bash

   pip install machlearn


-----

Example 1: k-Nearest Neighbors 
------------------------------

.. code-block:: python
   
   from machlearn import kNN
   kNN.demo("iris")


Selected Output:

.. code-block::

   This demo uses a public dataset of Fisher's Iris, which has a total of 150 samples from three species of Iris ('setosa', 'versicolor', 'virginica').
   The goal is to use 'the length and the width of the sepals and petals, in centimeters', to predict which species of Iris the sample belongs to.
   
   Using a grid search and a kNN classifier, the best hyperparameters were found as following:
      Step1: scaler: StandardScaler(with_mean=True, with_std=True);
      Step2: classifier: kNN_classifier(n_neighbors=12, weights='uniform', p=2.00, metric='minkowski').


|image_dataset_iris|
|image_kNN_iris_confusion_matrix|
   

.. |image_dataset_iris| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/datasets/iris/images/iris.jpg
   :width: 600px

.. |image_kNN_iris_confusion_matrix| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/kNN/images/iris_cm.png
   :width: 600px


-----

Example 2: Naive Bayes 
----------------------

.. code-block:: python
   
   from machlearn import naive_bayes as nb
   nb.demo(dataset="SMS_spam")


Selected Output:

.. code-block::

   This demo uses a public dataset of SMS spam, which has a total of 5574 messages = 4827 ham (legitimate) and 747 spam.
   The goal is to use 'term frequency in message' to predict whether the message is ham (class=0) or spam (class=1).

   Using a grid search and a multinomial naive bayes classifier, the best hyperparameters were found as following:
      Step1: Tokenizing text: CountVectorizer(analyzer = <_lemmas>, ngram_range = (1, 1));
      Step2: Transforming from occurrences to frequency: TfidfTransformer(use_idf = True).

   The top 2 terms with highest probability of a message being a spam (the classification is either spam or ham):
      "claim": 81.28%
      "prize": 80.24%
      "won": 76.29%

   Application example:
      - Message: "URGENT! We are trying to contact U. Todays draw shows that you have won a 2000 prize GUARANTEED. Call 090 5809 4507 from a landline. Claim 3030. Valid 12hrs only."
      - Probability of spam (class=1): 95.85%
      - Classification: spam


|image_SMS_spam_text_example|
|image_naive_bayes_confusion_matrix|


.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - |image_naive_bayes_ROC_curve| 
     - |image_naive_bayes_PR_curve| 


.. |image_SMS_spam_text_example| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/naive_bayes/images/SMS_spam_text_example.png
   :width: 600px

.. |image_naive_bayes_confusion_matrix| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/naive_bayes/images/demo_confusion_matrix.png
   :width: 600px

.. |image_naive_bayes_ROC_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/naive_bayes/images/demo_ROC_curve.png
   :width: 400px
   
.. |image_naive_bayes_PR_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/naive_bayes/images/demo_PR_curve.png
   :width: 400px


-----

Example 3: Decision Boundary Comparison (Classification with Two Features)
--------------------------------------------------------------------------

.. code-block:: python
   
   from machlearn import kNN
   kNN.demo("Social_Network_Ads")

   from machlearn import naive_bayes as nb
   nb.demo("Social_Network_Ads")

   from machlearn import SVM
   SVM.demo("Social_Network_Ads")
   
   from machlearn import decision_tree as DT
   DT.demo("Social_Network_Ads", classifier_func = "DT")

   from machlearn import logistic_regression
   logistic_regression.demo("Social_Network_Ads")


.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - |image_kNN_decision_boundary_testing_set|
     - |image_Gaussian_NB_decision_boundary_testing_set|
   * - |image_SVM_decision_boundary_testing_set|
     - |image_DT_decision_boundary_testing_set|
   * - |image_logistic_regression_decision_boundary_testing_set|
     -


.. |image_kNN_decision_boundary_testing_set| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/kNN/images/Social_Network_Ads_decision_boundary_testing_set.png
   :width: 400px

.. |image_Gaussian_NB_decision_boundary_testing_set| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/naive_bayes/images/demo_Social_Network_Ads_decision_boundary_testing_set.png
   :width: 400px

.. |image_SVM_decision_boundary_testing_set| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/SVM/images/Social_Network_Ads_decision_boundary_testing_set.png
   :width: 400px
   
.. |image_DT_decision_boundary_testing_set| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/decision_tree/images/Social_Network_Ads_DT_decision_boundary_testing_set.png
   :width: 400px

.. |image_logistic_regression_decision_boundary_testing_set| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/logistic_regression/images/Social_Network_Ads_decision_boundary_testing_set.png
   :width: 400px


-----

Example 4: Imbalanced Data
--------------------------

.. code-block:: python
   
   from machlearn import imbalanced_data
   imbalanced_data.demo()

Explanation:

.. code-block::

   To mitigate the problem associated with class imbalance, downsampling the majority class (y=0) to match the minority case (y=1).
   Note:
   - ROC curve is independent of class imbalance.
   - PR curve is indicative of the problem associated with class imbalance.


.. list-table::
   :widths: 25 25
   :header-rows: 1


   * - Extreme Imbalanced Data
     - Majority Downsampled to Match Minority
   * - |image_extreme_imbalanced_data_bar_chart|
     - |image_balanced_data_bar_chart|
   * - |image_extreme_imbalanced_data_confusion_matrix|
     - |image_balanced_data_confusion_matrix|
   * - |image_extreme_imbalanced_data_ROC_curve|
     - |image_balanced_data_ROC_curve|
   * - |image_extreme_imbalanced_data_PR_curve|
     - |image_balanced_data_PR_curve|


.. |image_extreme_imbalanced_data_bar_chart| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/imbalanced_data/images/extreme_imbalanced_data_bar_chart.png
   :width: 400px

.. |image_balanced_data_bar_chart| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/imbalanced_data/images/majority_downsampled_balanced_data_bar_chart.png
   :width: 400px

.. |image_extreme_imbalanced_data_confusion_matrix| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/imbalanced_data/images/extreme_imbalanced_data_logistic_regression_confusion_matrix.png
   :width: 400px

.. |image_balanced_data_confusion_matrix| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/imbalanced_data/images/majority_downsampled_balanced_data_logistic_regression_confusion_matrix.png
   :width: 400px

.. |image_extreme_imbalanced_data_ROC_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/imbalanced_data/images/extreme_imbalanced_data_logistic_regression_ROC_curve.png
   :width: 400px

.. |image_balanced_data_ROC_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/imbalanced_data/images/majority_downsampled_balanced_data_logistic_regression_ROC_curve.png
   :width: 400px

.. |image_extreme_imbalanced_data_PR_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/imbalanced_data/images/extreme_imbalanced_data_logistic_regression_PR_curve.png
   :width: 400px

.. |image_balanced_data_PR_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/imbalanced_data/images/majority_downsampled_balanced_data_logistic_regression_PR_curve.png
   :width: 400px


-----

module: model_evaluation
------------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "plot_confusion_matrix()", "plots the confusion matrix, along with key statistics, and returns accuracy"
   "plot_ROC_curve()", "plots the ROC (Receiver Operating Characteristic) curve, along with statistics"
   "plot_PR_curve()", "plots the precision-recall curve, along with statistics"
   "plot_ROC_and_PR_curves()", "plots both the ROC and the precision-recall curves, along with statistics"
   "demo()", "provides a demo of the major functions in this module"
   "demo_CV()", "provides a demo of cross validation in this module"


-----

module: datasets
----------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "public_dataset()", "returns a public dataset as specified (e.g., iris, SMS_spam, Social_Network_Ads)"


-----

module: kNN
-----------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "demo()", "provides a demo of selected functions in this module"


-----

module: naive_bayes
-------------------

.. csv-table::
   :header: "function/class", "description"
   :widths: 10, 20

   "naive_bayes_Bernoulli()", "when X are independent binary variables (e.g., whether a word occurs in a document or not)"
   "naive_bayes_multinomial()", "when X are independent discrete variables with 3+ levels (e.g., term frequency in the document)"
   "naive_bayes_Gaussian()", "when X are continuous variables"
   "demo()", "provides a demo of selected functions in this module"


-----

module: SVM
-----------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "demo()", "provides a demo of selected functions in this module"
   

-----

module: decision_tree
---------------------

.. csv-table::
   :header: "function/class", "description"
   :widths: 10, 20

   "decision_tree()", "decision tree classifier"
   "random_forest()", "random forest classifier"
   "bagging()", "bagging classifier"
   "AdaBoost()", "Adaptive Boosting classifier"
   "GBM()", "Gradient Boosting Machines classifier"
   "demo()", "provides a demo of selected functions in this module"


-----

module: neural_network
----------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "rnn()", "Recurrent neural network"
   "demo()", "provides a demo of selected functions in this module"


-----

module: logistic_regression
---------------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "LogisticReg_sklearn()", "solutions using sklearn"
   "LogisticReg_statsmodels()", "solutions using statsmodels"
   "demo()", "provides a demo of selected functions in this module"


-----

module: linear_regression
-------------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "Linear_regression()", "Linear_regression"
   "Linear_regression_normal_equation()", "Linear_regression_normal_equation"
   "Ridge_regression()", "Ridge_regression"
   "Lasso_regression()", "Lasso_regression"
   "demo()", "provides a demo of selected functions in this module"
   "demo_regularization()", "provides a demo of selected functions in this module"


-----

module: DSA
-----------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "demo()", "provides a demo of selected functions in this module"


-----

module: imbalanced_data
-----------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "demo()", "provides a demo of selected functions in this module"
