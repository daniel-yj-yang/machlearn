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
import numpy as np

class Multinomial_NB_classifier_from_scratch(object):
    # reference: https://geoffruddock.com/naive-bayes-from-scratch-with-numpy/

    def __init__(self, alpha=1.0, verbose=True):
        self.alpha = alpha # to avoid having zero probabilities for words not seen in our training sample.
        self.y_classes = None  # e.g., spam vs. no spam
        self.prob_y = None # Our prior belief in the probability of any randomly selected message belonging to a particular class
        self.prob_x_i_given_y = None # The likelihood of each word, conditional on message class.
        self.is_fitted = False
        self.verbose = verbose

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names=None):
        """
        X_train: a matrix of samples x features, such as documents (row) x words (col)
        """
        from sklearn.utils import check_X_y
        self.X_train, self.y_train = check_X_y(X_train, y_train)
        n_samples, n_features = X_train.shape

        if feature_names is None:
            self.feature_names = [f"word_{i}" for i in range(1,n_features+1)]
        else:
            self.feature_names = feature_names

        self.y_classes = np.unique(y_train)
        columns = [f"y={c}" for c in self.y_classes]

        X_train_by_y_class = np.array([X_train[y_train == this_y_class] for this_y_class in self.y_classes], dtype=object)
        self.prob_y = np.array([X_train_for_this_y_class.shape[0] / n_samples for X_train_for_this_y_class in X_train_by_y_class])
        if self.verbose:
            print(f"\nthe prior probability of y, before X is observed\nprior prob(y):\n{pd.DataFrame(self.prob_y.reshape(1,-1), columns=columns).to_string(index=False)}")

        # axis=0 means column-wise, axis=1 means row-wise
        self.X_train_colSum_by_y_class = np.array([ X_train_for_this_y_class.sum(axis=0) for X_train_for_this_y_class in X_train_by_y_class ]) + self.alpha
        self.prob_x_i_given_y = self.X_train_colSum_by_y_class / self.X_train_colSum_by_y_class.sum(axis=1).reshape(-1,1)
        if self.verbose:
            print(f"\nprob(word_i|y):\n{pd.concat([ pd.DataFrame(feature_names, columns=['word_i',]), pd.DataFrame(self.prob_x_i_given_y.T, columns = columns)], axis=1).to_string(index=False)}")

        self.is_fitted = True

        if self.verbose:
            self.predict_proba(self.X_train)

        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        p(y|X) = p(X|y)*p(y)/p(X)
        p(X|y) = p(x_1|y) * p(x_2|y) * ... * p(x_J|y)
        X: message (document), X_i: word
        """
        from sklearn.utils import check_array
        self.X_test = check_array(X_test)
        assert self.is_fitted, "model should be fitted first before predicting"

        #class_numerators = np.zeros(shape=(X_test.shape[0], self.prob_y.shape[0]))
        self.prob_X_given_y = np.zeros(shape=(X_test.shape[0], self.prob_y.shape[0]))

        # loop over each row to calcuate the posterior probability
        for row_index, this_x_sample in enumerate(X_test):
            feature_presence_columns = this_x_sample.astype(bool)
            prob_x_i_given_y_for_feature_present = self.prob_x_i_given_y[:, feature_presence_columns] ** this_x_sample[feature_presence_columns] # the "**" is likely a scaling factor to normalize prob_x_i
            # axis=0 means column-wise, axis=1 means row-wise
            self.prob_X_given_y[row_index] = (prob_x_i_given_y_for_feature_present).prod(axis=1)

        self.prob_X = (self.prob_X_given_y * self.prob_y).sum(axis=1).reshape(-1, 1)
        self.prob_y_given_X_test = self.prob_X_given_y * self.prob_y / self.prob_X  # the posterior probability of y, after X is observed
        assert (self.prob_y_given_X_test.sum(axis=1)-1 < 1e-9).all(), "each row should sum to 1"

        if self.verbose:
            columns = [f"y={c}" for c in self.y_classes]
            print(f"\nprob(X_message|y), where p(word_1|y) * p(word_2|y) * ... * p(word_J|y):\n{pd.DataFrame(self.prob_X_given_y, columns=columns).to_string(index=False)}")
            print(f"\nprob(X_message) across all possible y classes:\n{self.prob_X}")
            print(f"\nthe posterior prob of y after X is observed:\nprob(y|X) via p(y|X) = p(X|y)*p(y)/p(X):\n{pd.DataFrame(self.prob_y_given_X_test, columns=columns).to_string(index=False)}")

        return self.prob_y_given_X_test

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """ Predict class with highest probability """
        return self.predict_proba(X_test).argmax(axis=1)






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


def naive_bayes_Multinomial(*args, **kwargs):
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
                                    naive_bayes_Multinomial()),
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
        from ..datasets import public_dataset
        twenty_train = fetch_20newsgroups( #data_home=public_dataset("scikit_learn_data_path"),
            subset='train', categories=self.y_classes, random_state=self.random_state)
        twenty_test = fetch_20newsgroups( #data_home=public_dataset("scikit_learn_data_path"),
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


def demo_from_scratch():

    max_df = 1.0

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, _document_frequency
    vectorizer = CountVectorizer(max_df = max_df)
    X = document = ['BB AA', 'BB CC']
    y = ['spam', 'ham']
    print(f"1. document = {document}")
    transformed_data = vectorizer.fit_transform(X)
    term_frequency = transformed_data
    print(f"\n2. Term frequency (tf) (the number of times a word appears in the document):\n{pd.DataFrame(term_frequency.toarray(), columns = vectorizer.get_feature_names()).to_string(index=False)}")

    document_frequency = _document_frequency(term_frequency)
    document_frequency_divided_by_n_documents = np.divide(document_frequency, len(X))
    print(f"\n3a. Document frequency (df) (the number of times a word appears in the corpus):\n{pd.DataFrame(document_frequency.reshape(1,-1), columns = vectorizer.get_feature_names()).to_string(index=False)}")
    print(f"\n3b. Document frequency (df) / n_documents (this is where min_df and max_df could affect):\n{pd.DataFrame(document_frequency_divided_by_n_documents.reshape(1,-1), columns = vectorizer.get_feature_names()).to_string(index=False)}")

    # max_df: If float in range [0.0, 1.0], the parameter represents a proportion of documents
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df)
    transformed_data = tfidf_vectorizer.fit_transform(X)
    inverse_document_frequency = tfidf_vectorizer._tfidf._idf_diag
    #print(f"\n3. Inverse document frequency (adjust for the fact that some words appear more frequently in the corpus):\n{list(zip(tfidf_vectorizer.get_feature_names(), np.ravel(tfidf_vectorizer.idf_)))}")
    print(f"\n4a. Inverse document frequency (idf) (adjust for the fact that some words appear more frequently in the corpus):\n{pd.DataFrame(tfidf_vectorizer.idf_.reshape(1,-1), columns = tfidf_vectorizer.get_feature_names()).to_string(index=False)}")
    print(f"\n4b. Inverse document frequency (diag):\n{inverse_document_frequency.toarray()}")

    tf_times_idf = term_frequency * inverse_document_frequency
    print(f"\n5. Term frequency * Inverse document frequency (diag):\n{tf_times_idf.toarray()}")
    from sklearn.preprocessing import normalize
    normalized_tf_times_idf = normalize(tf_times_idf, norm = 'l2', axis=1) # axis=0 means column-wise, axis=1 means row-wise
    print(f"\n6.Document-wise normalized TF * IDF (each document has a unit length):\n{normalized_tf_times_idf.toarray()}")
    X_train = pd.DataFrame(transformed_data.toarray(), columns = tfidf_vectorizer.get_feature_names())
    print(f"\n7.Compared to transformed matrix from TfidfVectorizer() [should be the same]:\n{X_train.to_string(index=False)}")

    y_train = pd.DataFrame(y, columns = ['y',])
    print(f"\n8.y_train and X_train together:\n{pd.concat([y_train, X_train], axis=1).to_string(index=False)}")

    #################

    X_train = transformed_data.toarray()
    y_train = np.array(y)
    model_sklearn = naive_bayes_Multinomial()
    model_sklearn.fit(X_train, y_train)
    print(f"\nprob(y|X) from sklearn:\n{model_sklearn.predict_proba(X_train)}")

    model_from_scratch = Multinomial_NB_classifier_from_scratch()
    model_from_scratch.fit(X_train, y_train, feature_names=tfidf_vectorizer.get_feature_names())
    return model_from_scratch, X_train, y_train
    #print(f"\nprediction:{model_from_scratch.predict(X_test)}")

    #################

    # reference: https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
    # reference: https://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html#sphx-glr-auto-examples-applications-plot-out-of-core-classification-py
    y_classes = ['comp.graphics', 'sci.med']
    from sklearn.datasets import fetch_20newsgroups
    from ..datasets import public_dataset
    # no need to specify # data_home=public_dataset("scikit_learn_data_path"),
    twenty_train = fetch_20newsgroups( subset='train', categories=y_classes, random_state=1 )
    twenty_test  = fetch_20newsgroups( subset='test',  categories=y_classes, random_state=1 )
    
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(twenty_train.data)
    X_test =  vectorizer.transform(twenty_test.data)
    y_train, y_test = twenty_train.target, twenty_test.target

    model_sklearn = naive_bayes_Multinomial()
    model_sklearn.fit(X_train, y_train)
    print(f"\nprediction:{model_sklearn.predict(X_test)}")


