# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

# Some references:
# https://huggingface.co/docs/datasets/
# https://scikit-learn.org/stable/datasets/index.html#general-dataset-api

import io
#from zipfile import ZipFile
#import urllib.request
import csv

import pkgutil

import os
import numpy as np

def public_dataset(name=None):
    """
    name can be one of the following:
        - iris
        - SMS_spam
        - Social_Network_Ads
        - bank_note_authentication
        - marketing
        - Hitters
        - Gender
        - boston
        - Fashion_MNIST
        - nltk_data_path
        - scikit_learn_data_path

    Disclaimer:
        - The datasets are shared with the sole intention to provide the convenience of accessing publicly available datasets and reproducing/comparing results.
        - They are shared under a good-faith understanding that they are widely viewed and accepted as public-domain datasets.
        - If there is any misunderstanding, please contact the author.
        - The author does not own any of these datasets.
        - The readme in respective folder (or related Internet link) should be followed for citation/license requirements.
    """

    if name == "iris":
        import pandas as pd
        from sklearn import datasets
        iris = datasets.load_iris()
        dataset = pd.DataFrame(data = iris.data, columns = iris.feature_names)
        dataset['target'] = iris.target
        # iris.target_names  # y_classes = ['setosa', 'versicolor', 'virginica']
        print(f"Fisher's Iris is a publicly available dataset that consists of {len(iris.data)} samples from three species of Iris ('setosa', 'versicolor', 'virginica'), while four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.\n")
        return dataset

    #print(public_dataset.__doc__)
    if name == "SMS_spam":
        import pandas as pd
        # https://archive.ics.uci.edu/ml/datasets/sms+spam+collection (UCI Machine Learning Repository)
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/UCI_Machine_Learning_Repository/SMS_Spam_Collection/SMSSpamCollection.tsv")), sep='\t', quoting=csv.QUOTE_NONE, names=("label", "message"))
        n_spam = df['label'].value_counts()['spam']
        n_ham = df['label'].value_counts()['ham']
        print(f"SMS_spam is a publicly available dataset that has a total of {len(df)} messages = {n_ham} ham (legitimate) and {n_spam} spam.\n")
        return df
        #url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip")
        #df = pd.read_csv(ZipFile(io.BytesIO(url.read())).open('SMSSpamCollection'), sep='\t', quoting=csv.QUOTE_NONE, names=("label", "message"))

    if name == "Social_Network_Ads":
        import pandas as pd
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/Social_Network_Ads/Social_Network_Ads.csv")), encoding='utf8', sep=",")
        print("Social Network Ads is a publicly available dataset that can be used to determine what audience a car company should target in its ads in order to sell a SUV on a social network website.\n")
        return df
        #url = urllib.request.urlopen("https://github.com/daniel-yj-yang/machlearn/raw/master/machlearn/datasets/public/Social_Network_Ads/Social_Network_Ads.csv")
        #df = pd.read_csv(io.BytesIO(url.read()), encoding='utf8', sep=",")

    if name == "bank_note_authentication":
        # http://archive.ics.uci.edu/ml/datasets/banknote+authentication (UCI Machine Learning Repository)
        import pandas as pd
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/UCI_Machine_Learning_Repository/bank_note_authentication/data_banknote_authentication.txt")), header=None, encoding='utf8', sep=",")
        # 'variance of Wavelet Transformed image', 'skewness of Wavelet Transformed image', 'curtosis of Wavelet Transformed image', 'entropy of image'
        df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
        print(f"The dataset of bank note authentication is a publicly available dataset, where data were extracted from {len(df)} images (400x400 pixels, resolution of about 660 dpi) taken from {df['class'].value_counts()[0]} genuine and {df['class'].value_counts()[1]} forged banknote-like specimens. Wavelet Transform tool were used to extract features from images.\n")
        return df

    if name == "marketing":
        # https://cran.r-project.org/web/packages/datarium/datarium.pdf (GPL-2 License)
        import pandas as pd
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/R_datarium/marketing/marketing.csv")), header=0, encoding='utf8', sep=",")
        print(f"The dataset of marketing is a publicly available dataset from R-datarium, containing the impact of three advertising medias (youtube, facebook and newspaper) on sales. Data are the advertising budget in thousands of dollars along with the sales. The advertising experiment has been repeated 200 times.\n")
        return df

    if name == "Hitters":
        # https://cran.r-project.org/web/packages/ISLR/index.html (GPL-2 License)
        import pandas as pd
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/R_ISLR/Hitters/Hitters.csv")), header=0, encoding='utf8', sep=",")
        print(f"The dataset of Hitters is a publicly available dataset from R-ISLR, containing Major League Baseball Data from the 1986 and 1987 seasons.\n")
        return df

    if name == "Gender":
        # https://github.com/johnmyleswhite/ML_for_Hackers/blob/master/02-Exploration/data/01_heights_weights_genders.csv (FreeBSD License)
        import pandas as pd
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/ML_for_Hackers/01_heights_weights_genders.csv")), header=0, encoding='utf8', sep=",")
        print(f"The dataset of Gender is a publicly available dataset, containing 10000 heights and weights from 5000 males and 5000 females.\n")
        return df

    if name == "boston":
        import pandas as pd
        from sklearn.datasets import load_boston
        boston = load_boston()
        X = pd.DataFrame(data=boston.data, columns=boston.feature_names)
        y = pd.DataFrame(data=boston.target, columns=['MEDV'])
        df = pd.concat([y, X], axis=1)
        print("The dataset of boston is a publicly available dataset, including 506 cases of housing price in the area of Boston, Mass.")
        print("Here are the 13 X features:")
        print("1. CRIM - per capita crime rate by town")
        print("2. ZN - proportion of residential land zoned for lots over 25,000 sq.ft.")
        print("3. INDUS - proportion of non-retail business acres per town.")
        print("4. CHAS - Charles River dummy variable(1 if tract bounds river; 0 otherwise)")
        print("5. NOX - nitric oxides concentration (parts per 10 million)")
        print("6. RM - average number of rooms per dwelling")
        print("7. AGE - proportion of owner-occupied units built prior to 1940")
        print("8. DIS - weighted distances to five Boston employment centres")
        print("9. RAD - index of accessibility to radial highways")
        print("10. TAX - full-value property-tax rate per $10,000")
        print("11. PTRATIO - pupil-teacher ratio by town")
        print("12. B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")
        print("13. LSTAT - % lower status of the population")
        print("")
        print("Here is the y target variable:")
        print("14. MEDV - Median value of owner-occupied homes in $1000's\n")
        return [boston.data, boston.target, df]

    if name == "Fashion_MNIST":
        # this part of the code is modeled after https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
        import gzip
        path = os.path.dirname(__file__) + "/public/Fashion_MNIST"
        images_train_filepath = os.path.join(path, 'train-images-idx3-ubyte.gz')
        labels_train_filepath = os.path.join(path, 'train-labels-idx1-ubyte.gz')
        images_test_filepath  = os.path.join(path,  't10k-images-idx3-ubyte.gz')
        labels_test_filepath  = os.path.join(path,  't10k-labels-idx1-ubyte.gz')
        with gzip.open(labels_train_filepath, 'rb') as lbpath:
            labels_train = np.frombuffer(lbpath.read(),  dtype=np.uint8, offset=8)
        with gzip.open(images_train_filepath, 'rb') as imgpath:
            images_train = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels_train), 784)
        with gzip.open(labels_test_filepath,  'rb') as lbpath:
            labels_test  = np.frombuffer(lbpath.read(),  dtype=np.uint8, offset=8)
        with gzip.open(images_test_filepath,  'rb') as imgpath:
            images_test  = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels_test), 784)
        return images_train, labels_train, images_test, labels_test

    if name == 'nltk_data_path':
        return os.path.dirname(__file__) + "/public/nltk_data"

    if name == 'scikit_learn_data_path':
        return os.path.dirname(__file__) + "/public/scikit_learn_data"

    raise TypeError('recognizable dataset name is not provided')


