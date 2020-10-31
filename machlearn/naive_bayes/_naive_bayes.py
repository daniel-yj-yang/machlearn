# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause


#from ..datasets import public_dataset
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from textblob import TextBlob
import pandas as pd


#class naive_bayes_Bernoulli(BernoulliNB):
#    """
#    This class is used when X are independent binary variables (e.g., whether a word occurs in a document or not).
#    """
#    def __init__(self, *, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):
#        super().__init__(alpha=alpha, binarize=binarize, fit_prior=fit_prior, class_prior=class_prior)


#class naive_bayes_multinomial(MultinomialNB):
#    """
#    This class is used when X are independent discrete variables with 3+ levels (e.g., term frequency in the document).
#    """
#    # note: In Python 3, adding * to a function's signature forces calling code to pass every argument defined after the asterisk as a keyword argument
#    def __init__(self, *, alpha=1.0, fit_prior=True, class_prior=None): 
#        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)


#class naive_bayes_Gaussian(GaussianNB):
#    """
#    This class is used when X are continuous variables.
#    """
#    def __init__(self, *, priors=None, var_smoothing=1e-09):
#        super().__init__(priors=priors, var_smoothing=var_smoothing)


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


class _naive_bayes_demo():
    def __init__(self):
        self.X = None
        self.y = None
        self.y_classes = None
        self.test_size = 0.25
        self.classifier_grid = None
        self.random_state = 123
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_score = None

    def build_naive_bayes_Gaussian_pipeline(self):
        # create pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(steps=[('scaler',
                                    StandardScaler(with_mean=True, with_std=True)),
                                   ('classifier',
                                    naive_bayes_Gaussian()),
                                   ])
        # pipeline parameters to tune
        hyperparameters = {
            'scaler__with_mean': [True],
            'scaler__with_std': [True],
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
        print(
            "Training a Gaussian naive bayes pipeline, while tuning hyperparameters...\n")
        self.classifier_grid = grid.fit(self.X_train, self.y_train)
        print(
            f"Using a grid search and a Gaussian naive bayes classifier, the best hyperparameters were found as following:\n"
            f"Step1: scaler: StandardScaler(with_mean={repr(self.classifier_grid.best_params_['scaler__with_mean'])}, with_std={repr(self.classifier_grid.best_params_['scaler__with_std'])}).\n")

    def _lemmas(self, X):
        words = TextBlob(str(X).lower()).words
        return [word.lemma for word in words]

    def _tokens(self, X):
        return TextBlob(str(X)).words

    def build_naive_bayes_multinomial_pipeline(self):
        # create pipeline
        pipeline = Pipeline(steps=[('count_matrix_transformer',
                                    CountVectorizer(ngram_range=(1, 1), analyzer=self._tokens)),
                                   ('count_matrix_normalizer',
                                    TfidfTransformer(use_idf=True)),
                                   ('classifier',
                                    naive_bayes_multinomial()),
                                   ])
        # pipeline parameters to tune
        hyperparameters = {
            'count_matrix_transformer__ngram_range': ((1, 1), (1, 2)),
            'count_matrix_transformer__analyzer': (self._tokens, self._lemmas), # 'word', 
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
        print(
            "Training a multinomial naive bayes pipeline, while tuning hyperparameters...\n")

        #import nltk
        #nltk.download('punkt', quiet=True)
        #nltk.download('wordnet', quiet=True)

        #from ..datasets import public_dataset
        #import os
        #os.environ["NLTK_DATA"] = public_dataset("nltk_data_path")

        # see also: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
        # count_vect.fit_transform() in training vs. count_vect.transform() in testing
        self.classifier_grid = grid.fit(self.X_train, self.y_train)
        print(
            f"Using a grid search and a multinomial naive bayes classifier, the best hyperparameters were found as following:\n"
            f"Step1: Tokenizing text: CountVectorizer(ngram_range = {repr(self.classifier_grid.best_params_['count_matrix_transformer__ngram_range'])}, analyzer = {repr(self.classifier_grid.best_params_['count_matrix_transformer__analyzer'])});\n"
            f"Step2: Transforming from occurrences to frequency: TfidfTransformer(use_idf = {self.classifier_grid.best_params_['count_matrix_normalizer__use_idf']}).\n")


class _naive_bayes_demo_SMS_spam(_naive_bayes_demo):
    def __init__(self):
        super().__init__()
        self.y_classes = ('ham (y=0)', 'spam (y=1)')

    def getdata(self):
        from ..datasets import public_dataset
        data = public_dataset(name='SMS_spam')
        n_spam = data.loc[data.label == 'spam', 'label'].count()
        n_ham = data.loc[data.label == 'ham', 'label'].count()
        print(
            f"---------------------------------------------------------------------------------------------------------------------\n"
            f"This demo uses a public dataset of SMS spam, which has a total of {len(data)} messages = {n_ham} ham (legitimate) and {n_spam} spam.\n"
            f"The goal is to use 'term frequency in message' to predict whether a message is ham (class=0) or spam (class=1).\n")
        self.X = data['message']
        self.y = data['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def show_model_attributes(self):
        count_vect = self.classifier_grid.best_estimator_.named_steps['count_matrix_transformer']
        vocabulary_dict = count_vect.vocabulary_
        # clf = classifier_grid.best_estimator_.named_steps['classifier'] # clf = classifier fitted
        term_proba_df = pd.DataFrame({'term': list(
            vocabulary_dict), 'proba_spam': self.classifier_grid.predict_proba(vocabulary_dict)[:, 1]})
        term_proba_df = term_proba_df.sort_values(
            by=['proba_spam'], ascending=False)
        top_n = 10
        df = pd.DataFrame.head(term_proba_df, n=top_n)
        print(
            f"The top {top_n} terms with highest probability of a message being a spam (the classification is either spam or ham):")
        for term, proba_spam in zip(df['term'], df['proba_spam']):
            print(f"   \"{term}\": {proba_spam:4.2%}")

    def evaluate_model(self):
        self.y_pred = self.classifier_grid.predict(self.X_test)
        self.y_pred_score = self.classifier_grid.predict_proba(self.X_test)

        from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves
        plot_confusion_matrix(y_true=self.y_test, y_pred=self.y_pred,
                              y_classes=self.y_classes)
        plot_ROC_and_PR_curves(fitted_model=self.classifier_grid, X=self.X_test,
                               y_true=self.y_test, y_pred_score=self.y_pred_score[:, 1], y_pos_label='spam', model_name='Multinomial NB')

    def application(self):
        custom_message = "URGENT! We are trying to contact U. Todays draw shows that you have won a 2000 prize GUARANTEED. Call 090 5809 4507 from a landline. Claim 3030. Valid 12hrs only."
        custom_results = self.classifier_grid.predict([custom_message])[0]
        print(
            f"\nApplication example:\n- Message: \"{custom_message}\"\n- Probability of spam (class=1): {self.classifier_grid.predict_proba([custom_message])[0][1]:.2%}\n- Classification: {custom_results}\n")

    def run(self):
        """

        This function provides a demo of selected functions in this module using the SMS spam dataset.

        Required arguments:
            None

        """
        # Get data
        self.getdata()
        # Create and train a pipeline
        self.build_naive_bayes_multinomial_pipeline()
        # model attributes
        self.show_model_attributes()
        # model evaluation
        self.evaluate_model()
        # application example
        self.application()
        # return classifier_grid
        # return self.classifier_grid

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


class _naive_bayes_demo_20newsgroups(_naive_bayes_demo):
    def __init__(self):
        super().__init__()
        self.y_classes = sorted(
            ['soc.religion.christian', 'comp.graphics', 'sci.med'])

    def getdata(self):
        print(
            f"-------------------------------------------------------------------------------------------------------------------------------------\n"
            f"This demo uses a public dataset of 20newsgroup and uses {len(self.y_classes)} categories of them: {repr(self.y_classes)}.\n"
            f"The goal is to use 'term frequency in document' to predict which category a document belongs to.\n")
        from sklearn.datasets import fetch_20newsgroups
        #from ..datasets import public_dataset
        twenty_train = fetch_20newsgroups(  # data_home=public_dataset("scikit_learn_data_path"),
            subset='train', categories=self.y_classes, random_state=self.random_state)
        twenty_test = fetch_20newsgroups(  # data_home=public_dataset("scikit_learn_data_path"),
            subset='test', categories=self.y_classes, random_state=self.random_state)
        self.X_train = twenty_train.data
        self.y_train = twenty_train.target
        self.X_test = twenty_test.data
        self.y_test = twenty_test.target

    def show_model_attributes(self):
        # model attributes
        count_vect = self.classifier_grid.best_estimator_.named_steps['count_matrix_transformer']
        vocabulary_dict = count_vect.vocabulary_
        # clf = classifier_grid.best_estimator_.named_steps['classifier'] # clf = classifier fitted
        for i in range(len(self.y_classes)):
            term_proba_df = pd.DataFrame({'term': list(
                vocabulary_dict), 'proba': self.classifier_grid.predict_proba(vocabulary_dict)[:, i]})
            term_proba_df = term_proba_df.sort_values(
                by=['proba'], ascending=False)
            top_n = 10
            df = pd.DataFrame.head(term_proba_df, n=top_n)
            print(
                f"The top {top_n} terms with highest probability of a document being a {repr(self.y_classes[i])}:")
            for term, proba in zip(df['term'], df['proba']):
                print(f"   \"{term}\": {proba:4.2%}")

    def evaluate_model(self):
        # model evaluation
        self.y_pred = self.classifier_grid.predict(self.X_test)

        from ..model_evaluation import plot_confusion_matrix
        # the y_classes are in an alphabetic order
        plot_confusion_matrix(y_true=self.y_test,
                              y_pred=self.y_pred, y_classes=self.y_classes)

    def application(self):
        pass

    def run(self):
        """

        This function provides a demo of selected functions in this module using the 20 newsgroup dataset.

        It models after the tutorial https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

        Required arguments:
            None

        """
        # Get data
        self.getdata()
        # Create and train a pipeline
        self.build_naive_bayes_multinomial_pipeline()
        # model attributes
        self.show_model_attributes()
        # model evaluation
        self.evaluate_model()
        # application example
        self.application()
        # return classifier_grid
        # return self.classifier_grid


class _naive_bayes_demo_Social_Network_Ads(_naive_bayes_demo):
    def __init__(self):
        super().__init__()
        self.y_classes = ['not_purchased (y=0)', 'purchased (y=1)']

    def getdata(self):
        from ..datasets import public_dataset
        data = public_dataset(name='Social_Network_Ads')
        self.X = data[['Age', 'EstimatedSalary']].to_numpy()
        self.y = data['Purchased'].to_numpy()
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=123)

    def show_model_attributes(self):
        pass

    def evaluate_model(self):
        # model evaluation
        self.y_pred = self.classifier_grid.predict(self.X_test)
        self.y_pred_score = self.classifier_grid.predict_proba(self.X_test)

        from ..model_evaluation import plot_confusion_matrix, plot_ROC_and_PR_curves, visualize_classifier_decision_boundary_with_two_features
        plot_confusion_matrix(y_true=self.y_test,
                              y_pred=self.y_pred, y_classes=self.y_classes)
        plot_ROC_and_PR_curves(fitted_model=self.classifier_grid, X=self.X_test,
                               y_true=self.y_test, y_pred_score=self.y_pred_score[:, 1], y_pos_label=1, model_name="Gaussian NB")

        visualize_classifier_decision_boundary_with_two_features(
            self.classifier_grid, self.X_train, self.y_train, self.y_classes, title="Gaussian Naive Bayes / training set", X1_lab='Age', X2_lab='Estimated Salary')
        visualize_classifier_decision_boundary_with_two_features(
            self.classifier_grid, self.X_test,  self.y_test,  self.y_classes, title="Gaussian Naive Bayes / testing set",  X1_lab='Age', X2_lab='Estimated Salary')

    def application(self):
        pass

    def run(self):
        """

        This function provides a demo of selected functions in this module using the Social_Network_Ads dataset.

        Required arguments:
            None

        """
        # Get data
        self.getdata()
        # Create and train a pipeline
        self.build_naive_bayes_Gaussian_pipeline()
        # model attributes
        self.show_model_attributes()
        # model evaluation
        self.evaluate_model()
        # application example
        self.application()
        # return classifier_grid
        # return self.classifier_grid


def demo(dataset="SMS_spam"):
    """

    This function provides a demo of selected functions in this module.

    Required arguments:
        dataset: A string. Possible values: "SMS_spam", "20newsgroups", "Social_Network_Ads"

    """
    if dataset == "SMS_spam":
        nb_demo = _naive_bayes_demo_SMS_spam()
    elif dataset == "20newsgroups":
        nb_demo = _naive_bayes_demo_20newsgroups()
    elif dataset == "Social_Network_Ads":
        nb_demo = _naive_bayes_demo_Social_Network_Ads()
    else:
        raise TypeError(f"dataset [{dataset}] is not defined")
    return nb_demo.run()
