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


def __naive_bayes_multinomial_pipeline(X_train, y_train):
    # create pipeline
    pipeline = Pipeline(steps=[('count_matrix_transformer',
                                CountVectorizer(ngram_range=(1, 1), analyzer=__tokens)),
                               ('count_matrix_normalizer',
                                TfidfTransformer(use_idf=True)),
                               ('classifier',
                                naive_bayes_multinomial()),
                               ])
    # pipeline parameters to tune
    hyperparameters = {
        'count_matrix_transformer__ngram_range': ((1, 1), (1, 2)),
        'count_matrix_transformer__analyzer': ('word', __tokens, __lemmas),
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
    print("Training a multinomial naive bayes pipeline, while tuning hyperparameters...\n")

    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')

    # see also: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    # count_vect.fit_transform() in training vs. count_vect.transform() in testing
    classifier_grid = grid.fit(X_train, y_train)
    print(
        f"\nUsing a grid search, the best hyperparameters were found to be:\n"
        f"Step1: Tokenizing text: CountVectorizer(ngram_range = {repr(classifier_grid.best_params_['count_matrix_transformer__ngram_range'])}, analyzer = {repr(classifier_grid.best_params_['count_matrix_transformer__analyzer'])});\n"
        f"Step2: Transforming from occurrences to frequency: TfidfTransformer(use_idf = {classifier_grid.best_params_['count_matrix_normalizer__use_idf']}).\n")
    
    return classifier_grid

def _demo_SMS_spam():
    """

    This function provides a demo of selected functions in this module using the SMS spam dataset.

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
        f"The goal is to use 'term frequency in message' to predict whether a message is ham (class=0) or spam (class=1).\n")
    test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=test_size, random_state=123)

    # Build and train a pipeline
    classifier_grid = __naive_bayes_multinomial_pipeline(X_train, y_train)

    # model attributes
    count_vect = classifier_grid.best_estimator_.named_steps['count_matrix_transformer']
    vocabulary_dict = count_vect.vocabulary_
    # clf = classifier_grid.best_estimator_.named_steps['classifier'] # clf = classifier fitted
    term_proba_df = pd.DataFrame({'term': list(
        vocabulary_dict), 'proba_spam': classifier_grid.predict_proba(vocabulary_dict)[:, 1]})
    term_proba_df = term_proba_df.sort_values(
        by=['proba_spam'], ascending=False)
    top_n = 10
    df = pd.DataFrame.head(term_proba_df, n=top_n)
    print(f"The top {top_n} terms with highest probability of a message being a spam (the classification is either spam or ham):")
    for term, proba_spam in zip(df['term'], df['proba_spam']):
        print(f"   \"{term}\": {proba_spam:4.2%}")

    # model evaluation
    y_pred = classifier_grid.predict(X_test)
    y_score = classifier_grid.predict_proba(X_test)

    from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, y_classes=(
        'ham (y=0)', 'spam (y=1)'))
    plot_ROC_and_PR_curves(fitted_model=classifier_grid, X=X_test,
                           y_true=y_test, y_pred_score=y_score[:, 1], y_pos_label='spam', model_name='Multinomial NB')

    # application example
    custom_message = "URGENT! We are trying to contact U. Todays draw shows that you have won a 2000 prize GUARANTEED. Call 090 5809 4507 from a landline. Claim 3030. Valid 12hrs only."
    custom_results = classifier_grid.predict([custom_message])[0]
    print(
        f"\nApplication example:\n- Message: \"{custom_message}\"\n- Probability of class=1 (spam): {classifier_grid.predict_proba([custom_message])[0][1]:.2%}\n- Classification: {custom_results}\n")

    return classifier_grid

    # import numpy as np
    # from sklearn.utils import shuffle

    # True Positive
    #X_test_subset = X_test[y_test == 'spam']
    #y_pred_array = classifier_grid.predict( X_test_subset )
    #X_test_subset.loc[[ X_test_subset.index[ shuffle(np.where(y_pred_array == 'spam')[0], n_samples=1, random_state=1234)[0] ] ]]

    # False Negative
    #X_test_subset = X_test[y_test == 'spam']
    #y_pred_array = classifier_grid.predict( X_test_subset )
    #X_test_subset.loc[[ X_test_subset.index[ shuffle(np.where(y_pred_array == 'ham')[0], n_samples=1, random_state=1234)[0] ] ]]

    # False Positive
    #X_test_subset = X_test[y_test == 'ham']
    #y_pred_array = classifier_grid.predict( X_test_subset )
    #X_test_subset.loc[[ X_test_subset.index[ shuffle(np.where(y_pred_array == 'spam')[0], n_samples=1, random_state=1234)[0] ] ]]

    # True Negative
    #X_test_subset = X_test[y_test == 'ham']
    #y_pred_array = classifier_grid.predict( X_test_subset )
    #X_test_subset.loc[[ X_test_subset.index[ shuffle(np.where(y_pred_array == 'ham')[0], n_samples=1, random_state=123)[0] ] ]]


def _demo_20newsgroup():
    """

    This function provides a demo of selected functions in this module using the 20 newsgroup dataset.

    It models after the tutorial https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

    Required arguments:
        None

    """
    categories = sorted(['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med'])
    print(
        f"This demo uses a public dataset of 20newsgroup and uses {len(categories)} categories of them: {repr(categories)}.\n"
        f"The goal is to use 'term frequency in document' to predict which category a document belongs to.\n")

    from sklearn.datasets import fetch_20newsgroups
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, random_state=123)
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, random_state=123)

    # Build and train a pipeline
    classifier_grid = __naive_bayes_multinomial_pipeline(X_train=twenty_train.data, y_train=twenty_train.target)

    # model attributes
    count_vect = classifier_grid.best_estimator_.named_steps['count_matrix_transformer']
    vocabulary_dict = count_vect.vocabulary_
    # clf = classifier_grid.best_estimator_.named_steps['classifier'] # clf = classifier fitted
    for i in len(categories):
        term_proba_df = pd.DataFrame({'term': list(
            vocabulary_dict), 'proba': classifier_grid.predict_proba(vocabulary_dict)[:, i]})
        term_proba_df = term_proba_df.sort_values(
            by=['proba'], ascending=False)
        top_n = 10
        df = pd.DataFrame.head(term_proba_df, n=top_n)
        print(f"The top {top_n} terms with highest probability of a document being a {repr(categories[i])}:")
        for term, proba in zip(df['term'], df['proba']):
            print(f"   \"{term}\": {proba:4.2%}")

    # model evaluation
    y_pred = classifier_grid.predict(twenty_test.data)

    from ..model_evaluation import plot_confusion_matrix
    plot_confusion_matrix(y_true=twenty_test.target, y_pred=y_pred, y_classes=categories) # the y_classes are in an alphabetic order

    return classifier_grid

def demo(dataset="SMS_spam"):
    """

    This function provides a demo of selected functions in this module.

    Required arguments:
        dataset: A string. Possible values: "SMS_spam", "20newsgroup".

    """
    if dataset == "SMS_spam":
        return _demo_SMS_spam()
    if dataset == "20newsgroup":
        return _demo_20newsgroup()
    raise TypeError(f"dataset [{dataset}] not defined")
