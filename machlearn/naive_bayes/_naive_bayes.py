# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause


from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from textblob import TextBlob
import pandas as pd
import csv

from io import BytesIO
from zipfile import ZipFile
import urllib.request

import numpy as np
from sklearn.utils import shuffle


def naive_bayes_Bernoulli(*args, **kwargs):
    """

    This function is used when X are independent binary variables (e.g., whether a word occurs in a document or not).

    """
    return BernoulliNB(*args, **kwargs)


def naive_bayes_multinomial(*args, **kwargs):
    """

    This function is used when X are independent discrete variables with 3+ levels (e.g., term frequency in the document).

    """
    return MultinomialNB(*args, **kwargs)


def naive_bayes_Gaussian(*args, **kwargs):
    """

    This function is used when X are continuous variables.

    """
    return GaussianNB(*args, **kwargs)


def __lemmas(X):
    words = TextBlob(str(X).lower()).words
    return [word.lemma for word in words]


def __tokens(X):
    return TextBlob(str(X)).words


def demo():
    """

    This function provides a demo of selected functions in this module.

    Required arguments:
        None
        
    """
    url = urllib.request.urlopen(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip")
    data = pd.read_csv(ZipFile(BytesIO(url.read())).open(
        'SMSSpamCollection'), sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])
    n_spam = data.loc[data.label == 'spam', 'label'].count()
    n_ham = data.loc[data.label == 'ham', 'label'].count()
    print(
        f"This demo uses a public dataset of SMS spam, which has a total of {len(data)} messages = {n_ham} ham (legitimate) and {n_spam} spam.\n"
        f"The goal is to use 'term frequency in the message' to predict whether the message is ham (class=0) or spam (class=1).\n")
    test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=test_size, random_state=123)
    # create pipeline
    pipeline = Pipeline(steps=[('count_matrix_transformer',
                                CountVectorizer(analyzer=__tokens)),
                               ('count_matrix_normalizer',
                                TfidfTransformer(use_idf=True)),
                               ('classifier',
                                naive_bayes_multinomial()),
                               ])
    # pipeline parameters to tune
    hyperparameters = {
        'count_matrix_transformer__analyzer': (__tokens, __lemmas),
        'count_matrix_normalizer__use_idf': (True, False),
    }
    grid = GridSearchCV(
        pipeline,
        hyperparameters,  # parameters to tune via cross validation
        refit=True,       # fit using all data, on the best detected classifier
        n_jobs=-1,
        scoring='accuracy',
        cv=5,
    )
    # train
    print("Training a multinomial naive bayes model, while tuning hyperparameters...\n")

    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')

    classifier = grid.fit(X_train, y_train)
    print(
        f"Using test_size = {test_size}, the best hyperparameters for a multinomial NB model were found to be:\n"
        f"Step1: Convert from text to count matrix = CountVectorizer(analyzer = {classifier.best_params_['count_matrix_transformer__analyzer'].__name__});\n"
        f"Step2: Transform count matrix to tf-idf = TfidfTransformer(use_idf = {classifier.best_params_['count_matrix_normalizer__use_idf']}).\n")
    # model evaluation
    y_pred = classifier.predict(X_test)
    y_score = classifier.predict_proba(X_test)

    from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves
    accuracy = plot_confusion_matrix(y_true=y_test, y_pred=y_pred, y_classes=(
        'ham (y=0)', 'spam (y=1)'))
    plot_ROC_and_PR_curves(fitted_model=classifier, X=X_test,
                           y_true=y_test, y_pred_score=y_score[:, 1], y_pos_label='spam', model_name='Multinomial NB')
    # application example
    custom_message = "URGENT! We are trying to contact U. Todays draw shows that you have won a 2000 prize GUARANTEED. Call 090 5809 4507 from a landline. Claim 3030. Valid 12hrs only."
    custom_results = classifier.predict([custom_message])[0]
    print(
        f"Application example:\n- Message: \"{custom_message}\"\n- Probability of class=1 (spam): {classifier.predict_proba([custom_message])[0][1]:.2%}\n- Classification: {custom_results}\n")

    # True Positive
    #X_test_subset = X_test[y_test == 'spam']
    #y_pred_array = classifier.predict( X_test_subset )
    #X_test_subset.loc[[ X_test_subset.index[ shuffle(np.where(y_pred_array == 'spam')[0], n_samples=1, random_state=1234)[0] ] ]]

    # False Negative
    #X_test_subset = X_test[y_test == 'spam']
    #y_pred_array = classifier.predict( X_test_subset )
    #X_test_subset.loc[[ X_test_subset.index[ shuffle(np.where(y_pred_array == 'ham')[0], n_samples=1, random_state=1234)[0] ] ]]

    # False Positive
    #X_test_subset = X_test[y_test == 'ham']
    #y_pred_array = classifier.predict( X_test_subset )
    #X_test_subset.loc[[ X_test_subset.index[ shuffle(np.where(y_pred_array == 'spam')[0], n_samples=1, random_state=1234)[0] ] ]]

    # True Negative
    #X_test_subset = X_test[y_test == 'ham']
    #y_pred_array = classifier.predict( X_test_subset )
    #X_test_subset.loc[[ X_test_subset.index[ shuffle(np.where(y_pred_array == 'ham')[0], n_samples=1, random_state=123)[0] ] ]]
