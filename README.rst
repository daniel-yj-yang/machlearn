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
A Simple and Powerful Machine Learning Python Library
=====================================================

Install
-------

.. code-block:: bash

   pip install machlearn

-----

Example 1: Naive Bayes 
----------------------

.. code-block:: python
   
   from machlearn import naive_bayes as nb
   nb.demo(dataset="SMS_spam")


Selected Output:

.. code-block::

   This demo uses a public dataset of SMS spam, which has a total of 5574 messages = 747 spam and 4827 ham (legitimate).
   The goal is to use 'term frequency in message' to predict whether the message is ham (class=0) or spam (class=1).

   Using a grid search and a multinomial naive bayes classifier, the best hyperparameters were found as following:
      Step1: Tokenizing text: CountVectorizer(analyzer = 'word', ngram_range = (1, 1));
      Step2: Transforming from occurrences to frequency: TfidfTransformer(use_idf = True).

   The top 2 terms with highest probability of a message being a spam (the classification is either spam or ham):
      "claim": 80.73%
      "prize": 80.06%

   Application example:
      - Message: "URGENT! We are trying to contact U. Todays draw shows that you have won a 2000 prize GUARANTEED. Call 090 5809 4507 from a landline. Claim 3030. Valid 12hrs only."
      - Probability of class=1 (spam): 98.32%
      - Classification: spam


.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - |image_naive_bayes_confusion_matrix|
     -

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - |image_naive_bayes_ROC_curve| 
     - |image_naive_bayes_PR_curve| 

.. |image_naive_bayes_confusion_matrix| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/naive_bayes/images/demo_confusion_matrix.png
   :width: 400px

.. |image_naive_bayes_ROC_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/naive_bayes/images/demo_ROC_curve.png
   :width: 400px
   
.. |image_naive_bayes_PR_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/naive_bayes/images/demo_PR_curve.png
   :width: 400px


-----

Example 2: k-Nearest Neighbors 
------------------------------

.. code-block:: python
   
   from machlearn import kNN
   kNN.demo("Social_Network_Ads")


Selected Output:

.. code-block::

   This demo uses a public dataset of Social Network Ads, which is used to determine what audience a car company should target in its ads in order to sell a SUV on a social network website.
   
   Using a grid search and a kNN classifier, the best hyperparameters were found as following:
      Step1: scaler: StandardScaler(with_mean=True, with_std=True);
      Step2: classifier: kNN_classifier(n_neighbors=8, weights='uniform', p=1.189207115002721, metric='minkowski').


.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - |image_kNN_confusion_matrix|
     - |image_kNN_decision_boundary_testing_set|

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - |image_kNN_ROC_curve| 
     - |image_kNN_PR_curve| 

.. |image_kNN_confusion_matrix| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/kNN/images/Social_Network_Ads_cm.png
   :width: 400px

.. |image_kNN_ROC_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/kNN/images/Social_Network_Ads_ROC_curve.png
   :width: 400px
   
.. |image_kNN_PR_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/kNN/images/Social_Network_Ads_PR_curve.png
   :width: 400px


-----

Example 3: Decision Boundary Comparison
---------------------------------------

.. code-block:: python
   
   from machlearn import kNN
   kNN.demo("Social_Network_Ads")

   from machlearn import naive_bayes as nb
   nb.demo("Social_Network_Ads")

   from machlearn import SVM
   SVM.demo("Social_Network_Ads")


.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - |image_kNN_decision_boundary_testing_set|
     - |image_Gaussian_NB_decision_boundary_testing_set|
   * - |image_SVM_decision_boundary_testing_set|
     -


.. |image_kNN_decision_boundary_testing_set| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/kNN/images/Social_Network_Ads_decision_boundary_testing_set.png
   :width: 400px

.. |image_Gaussian_NB_decision_boundary_testing_set| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/naive_bayes/images/demo_Social_Network_Ads_decision_boundary_testing_set.png
   :width: 400px

.. |image_SVM_decision_boundary_testing_set| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/SVM/images/Social_Network_Ads_decision_boundary_testing_set.png
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

-----

module: naive_bayes
-------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "naive_bayes_Bernoulli()", "when X are independent binary variables (e.g., whether a word occurs in a document or not)"
   "naive_bayes_multinomial()", "when X are independent discrete variables with 3+ levels (e.g., term frequency in the document)"
   "naive_bayes_Gaussian()", "when X are continuous variables"
   "demo()", "provides a demo of selected functions in this module"

-----

module: kNN
-----------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

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

module: decision_tree
---------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "boost()", "Boosting"
   "demo()", "provides a demo of selected functions in this module"
