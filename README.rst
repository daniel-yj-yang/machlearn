.. -*- mode: rst -*-

|BuildTest|_ |PythonVersion|_ |PyPi|_ |Downloads|_ |License|_

.. |BuildTest| image:: https://travis-ci.com/daniel-yj-yang/machlearn.svg?branch=master
.. _BuildTest: https://travis-ci.com/daniel-yj-yang/machlearn

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue

.. |PyPi| image:: https://badge.fury.io/py/machlearn.svg
.. _PyPi: https://pypi.python.org/pypi/machlearn

.. |Downloads| image:: https://pepy.tech/badge/machlearn
.. _Downloads: https://pepy.tech/project/machlearn

.. |License| image:: https://img.shields.io/pypi/l/machlearn
.. _License: https://pypi.python.org/pypi/machlearn


===============================
Machine Learning Python Library
===============================

Install
-------

.. code-block:: bash

   pip install machlearn

-----

Example 1: Model evaluation
---------------------------

.. code-block:: python
   
   from machlearn import model_evaluation as me
   me.demo()


Selected Output:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - |image_demo_confusion_matrix|
     -

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - |image_demo_ROC_curve| 
     - |image_demo_PR_curve| 

.. |image_demo_confusion_matrix| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/model_evaluation/images/demo_confusion_matrix.png
   :width: 400px

.. |image_demo_ROC_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/model_evaluation/images/demo_ROC_curve.png
   :width: 400px
   
.. |image_demo_PR_curve| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/model_evaluation/images/demo_PR_curve.png
   :width: 400px


-----

Example 2: Naive Bayes 
----------------------

.. code-block:: python
   
   from machlearn import naive_bayes as nb
   nb.demo()


Selected Output:

.. code-block::

   This demo uses a public dataset of SMS spam, which has a total of 5574 messages = 747 spam and 4827 ham (legitimate).
   The goal is to use 'term frequency in the message' to predict whether the message is ham (class=0) or spam (class=1).

   Using test_size = 0.25 and training a multinomial naive bayes model, the best hyperparameters were found to be:
      Step1: Convert from text to count matrix = CountVectorizer(analyzer = __lemmas);
      Step2: Transform count matrix to tf-idf = TfidfTransformer(use_idf = True).

   Application example:
      - Message: "URGENT! We are trying to contact U. Todays draw shows that you have won a 2000 prize GUARANTEED. Call 090 5809 4507 from a landline. Claim 3030. Valid 12hrs only."
      - Probability of class=1 (spam): 95.85%
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
