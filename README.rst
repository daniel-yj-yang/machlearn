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
   kNN.demo_from_scratch("iris")


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
   nb.demo_from_scratch()
   nb.demo(dataset="SMS_spam")


Selected Output from nb.demo(dataset="SMS_spam"):

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

   from machlearn import logistic_regression as logreg
   logreg.demo("Social_Network_Ads")

   from machlearn import neural_network as NN
   NN.demo("Social_Network_Ads")

   from machlearn import ensemble
   ensemble.demo("Social_Network_Ads")


.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - |image_kNN_decision_boundary_testing_set|
     - |image_Gaussian_NB_decision_boundary_testing_set|
   * - |image_SVM_decision_boundary_testing_set|
     - |image_DT_decision_boundary_testing_set|
   * - |image_logistic_regression_decision_boundary_testing_set|
     - |image_NN_MLP_decision_boundary_testing_set|
   * - |image_RFC_decision_boundary_testing_set|
     - |image_GBM_decision_boundary_testing_set|


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

.. |image_NN_MLP_decision_boundary_testing_set| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/neural_network/images/Social_Nework_Ads_MLP_decision_boundary_testing_set.png
   :width: 400px

.. |image_RFC_decision_boundary_testing_set| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/ensemble/images/Social_Network_Ads_RFC_decision_boundary_testing_set.png
   :width: 400px

.. |image_GBM_decision_boundary_testing_set| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/ensemble/images/Social_Network_Ads_GBM_decision_boundary_testing_set.png
   :width: 400px


-----

Example 4: Imbalanced Data
--------------------------

.. code-block:: python
   
   from machlearn import imbalanced_data
   imbalanced_data.demo()

Summary of output:

.. code-block::

   To mitigate the problem associated with class imbalance, downsampling the majority class (y=0) to match the minority case (y=1).
   
   These are insensitive to class imbalance:
   - Area Under ROC curve
   - Geometric mean
   - Matthew's Correlation Coefficient
   - Recall, TPR
   - Specificity, 1-FPR

   These are sensitive to class imbalance:
   - Area Under PR curve
   - Accuracy
   - F1 score
   - Precision


.. list-table::
   :widths: 25 25
   :header-rows: 1


   * - Extreme Imbalanced Data
     - Majority Downsampled to Match Minority Class
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

Example 5: Regularization
-------------------------

.. code-block:: python
   
   from machlearn import linear_regression as linreg
   linreg.demo_regularization()

Summary of output:

.. code-block::

   Issues: (a) high multicollinearity and (b) too many features; these lead to overfitting and poor generalization.
   - After L2 Regularization (Ridge regression), reduced variance among the coefficient estimates [more robust/stable estimates], and better R-squared and lower RMSE with the testing set [better generalization]
   - After L1 Regularization (Lasso regression), coefficient estimates becoming 0 for relatively trivial features [a simpler model], and better R-squared and lower RMSE with the testing set [better generalization]


-----

Example 6: Gradient Descent
---------------------------

.. code-block:: python
   
   from machlearn import gradient_descent as GD
   GD.demo("Gender")

Summary of output:

.. code-block::

   This example uses a batch gradient descent (BGD) procedure, a cost function of logistic regression, 30,000 # iterations, a learning rate of 0.00025, and with Male (1, 0) as the target.
   - Theta estimates of [const, Height (inch), Weight (lbs)]: [-0.00977953, -0.4779923, 0.19667817]
   - Compared to estimates from statsmodels ([0.69254314, -0.49262002, 0.19834042]), the estimates associated with Height and Weight are very close
   - Accuracy of prediction:  0.919


.. list-table::
   :widths: 25 25
   :header-rows: 1


   * - Descriptive statistics
     - Batch Gradient Descent Training Loss vs. Epoch
   * - |image_Gender_pairplot|
     - |image_Gender_batch_gradient_descent_training_loss_plot|


|image_Gender_batch_gradient_descent_training_cost_vs_theta_plot|


.. |image_Gender_pairplot| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/gradient_descent/images/Gender_pairplot.png
   :width: 400px

.. |image_Gender_batch_gradient_descent_training_loss_plot| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/gradient_descent/images/Gender_BGD_training_loss_history.png
   :width: 400px

.. |image_Gender_batch_gradient_descent_training_cost_vs_theta_plot| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/gradient_descent/images/Gender_BGD_training_cost_vs_theta.png
   :width: 600px


-----

Example 7: Decision Tree
------------------------

.. code-block:: python
   
   from machlearn import decision_tree as DT
   DT.demo()
   DT.demo_from_scratch(question_type="regression") # dataset='boston'
   DT.demo_from_scratch(question_type="classification") # dataset='Social_Network_Ads', X=not scaled, criterion=entropy, max_depth=2


Summary of output:

.. code-block::

   - DT.demo_from_scratch(question_type="regression") uses decision_tree_regressor_from_scratch()
   - DT.demo_from_scratch(question_type="classification") provides results essentially identical to the tree graph below.


|image_Social_Networks_Ad_DT_notscaled_entropy_maxdepth=2|


.. |image_Social_Networks_Ad_DT_notscaled_entropy_maxdepth=2| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/decision_tree/images/Social_Network_Ads_DT_notscaled_entropy_max_depth=2_tree_graph.png
   :width: 800px


-----

Example 8: Ensemble Methods
---------------------------

.. code-block:: python
   
   from machlearn import ensemble
   ensemble.demo()
   ensemble.demo("Social_Network_Ads")
   ensemble.demo("boston")

Summary of output: 

.. code-block::

   - These demos call the following functions developed from scratch and reflect the inner workings of them:
   * random_forest_classifier_from_scratch();
   * adaptive_boosting_classifier_from_scratch();
   * gradient_boosting_regressor_from_scratch() (see training history plot below): R_squared = 0.753, RMSE = 4.419



|image_boston_GBM_loss_history_plot|


.. |image_boston_GBM_loss_history_plot| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/ensemble/images/boston_gradient_boosting_regressor_from_scratch_loss_vs_epoch_history_plot.png
   :width: 400px


-----

Example 9: Assumption Testing
-----------------------------

.. code-block:: python
   
   from machlearn import linear_regression as linreg
   linreg.demo_assumption_test()

Summary of output: 

.. code-block::

   The assumptions of linear regression include (1) linear relationship between X and y, (2) I.I.D. of the residuals (residuals are independently and identically distributed as normal), (3) little or no multicollineaity if multiple IVs.


Selected output:

.. list-table::
   :widths: 25 30
   :header-rows: 0

   * - |image_linreg_assumption_test_linearity|
     - |image_linreg_assumption_test_homoscedasticity|


.. |image_linreg_assumption_test_linearity| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/linear_regression/images/assumption_test_linearity.png
   :width: 400px

.. |image_linreg_assumption_test_homoscedasticity| image:: https://github.com/daniel-yj-yang/machlearn/raw/master/examples/linear_regression/images/assumption_test_homoscedasticity.png
   :width: 500px


-----

module: model_evaluation
------------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "plot_ROC_and_PR_curves()", "plots both the ROC and the precision-recall curves, along with statistics"
   "plot_ROC_curve()", "plots the ROC (Receiver Operating Characteristic) curve, along with statistics"
   "plot_PR_curve()", "plots the precision-recall curve, along with statistics"
   "plot_confusion_matrix()", "plots the confusion matrix, along with key statistics, and returns accuracy"
   "demo_CV()", "provides a demo of cross validation in this module"
   "demo()", "provides a demo of the major functions in this module"


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

   "kNN_classifier_from_scratch()", "kNN classifier developed from scratch"
   "demo_from_scratch()", "provides a demo of selected functions in this module"
   "demo()", "provides a demo of selected functions in this module"


-----

module: naive_bayes
-------------------

.. csv-table::
   :header: "class/function", "description"
   :widths: 10, 20

   "Multinomial_NB_classifier_from_scratch()", "Multinomial NB classifier developed from scratch"
   "demo_from_scratch()", "provides a demo of selected functions in this module" 
   "Gaussian_NB_classifier()", "when X are continuous variables"
   "Multinomial_NB_classifier()", "when X are independent discrete variables with 3+ levels (e.g., term frequency in the document)"
   "Bernoulli_NB_classifier()", "when X are independent binary variables (e.g., whether a word occurs in a document or not)"
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
   :header: "class/function", "description"
   :widths: 10, 20

   "decision_tree_regressor_from_scratch()", "decision tree regressor developed from scratch"
   "decision_tree_classifier_from_scratch()", "decision tree classifier developed from scratch"
   "demo_from_scratch()", "provides a demo of selected functions in this module"
   "decision_tree_regressor()", "decision tree regressor"
   "decision_tree_classifier()", "decision tree classifier"
   "demo()", "provides a demo of selected functions in this module"


-----

module: neural_network
----------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "multi_layer_perceptron_classifier()", "multi-layer perceptron (MLP) classifier"
   "rnn()", "recurrent neural network"
   "demo()", "provides a demo of selected functions in this module"


-----

module: logistic_regression
---------------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "logistic_regression_sklearn()", "solutions using sklearn"
   "logistic_regression_statsmodels()", "solutions using statsmodels"
   "demo()", "provides a demo of selected functions in this module"


-----

module: linear_regression
-------------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "assumption_test()", "tests the assumptions of linear regression"
   "lasso_regression()", "lasso_regression"
   "ridge_regression()", "ridge_regression"
   "linear_regression_normal_equation()", "linear_regression_normal_equation"
   "linear_regression()", "linear_regression"
   "demo()", "provides a demo of selected functions in this module"
   "demo_regularization()", "provides a demo of selected functions in this module"
   "demo_assumption_test()", "provides a demo of selected functions in this module"


-----

module: DSA
-----------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "demo()", "provides a demo of selected functions in this module"


-----

module: stats
-------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "demo()", "provides a demo of selected functions in this module"


-----

module: pipeline
----------------

.. csv-table::
   :header: "class/function", "description"
   :widths: 10, 20

   "demo()", "provides a demo of selected functions in this module"
   

-----

module: imbalanced_data
-----------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "demo()", "provides a demo of selected functions in this module"


-----

module: decomposition
---------------------

.. csv-table::
   :header: "function", "description"
   :widths: 10, 20

   "demo()", "provides a demo of selected functions in this module"


-----

module: gradient_descent
------------------------

.. csv-table::
   :header: "class/function", "description"
   :widths: 10, 20

   "logistic_regression_BGD_classifier()", "logistic_regression_BGD_classifier class"
   "batch_gradient_descent()", "batch_gradient_descent class"
   "demo()", "provides a demo of selected functions in this module"


-----

module: ensemble
----------------

.. csv-table::
   :header: "class/function", "description"
   :widths: 10, 20

   "gradient_boosting_regressor_from_scratch()", "gradient boosting regressor developed from scratch"
   "adaptive_boosting_classifier_from_scratch()", "adaptive boosting classifier developed from scratch"
   "random_forest_classifier_from_scratch()", "random forest classifier developed from scratch"
   "bagging_classifier_from_scratch()", "bagging classifier developed from scratch"
   "gradient_boosting_classifier()", "gradient boosting classifier"
   "adaptive_boosting_classifier()", "adaptive boosting classifier"
   "random_forest_classifier()", "random forest classifier"
   "bagging_classifier()", "bagging classifier"
   "voting_classifier()", "voting classifier"
   "demo()", "provides a demo of selected functions in this module"
   