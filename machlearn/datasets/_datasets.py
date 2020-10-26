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

import pandas as pd

def public_dataset(name=None):
    """
    name can be one of the following:
        - iris
        - SMS_spam
        - Social_Network_Ads
        - Fashion_MNIST
        - nltk_data_path
        - scikit_learn_data_path

    Disclaimer:
        - The datasets are shared with the sole intention of providing the convenience to access public datasets and reproduce/compare results.
        - They are shared under a good-faith understanding that they are widely viewed and accepted as public-domain datasets.
        - If there is any misunderstanding, please contact the author.
        - The the author does not own any of these datasets.
        - The readme in respective folder (or related Internet link) should be followed for citation/license requirements.
    """

    if name == 'iris':
        from sklearn import datasets
        iris = datasets.load_iris()
        dataset = pd.DataFrame(data = iris.data, columns = iris.feature_names)
        dataset['target'] = iris.target
        return dataset

    #print(public_dataset.__doc__)
    if name == 'SMS_spam':
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/SMS_Spam_Collection/SMSSpamCollection.tsv")), sep='\t', quoting=csv.QUOTE_NONE, names=("label", "message"))
        return df
        #url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip")
        #df = pd.read_csv(ZipFile(io.BytesIO(url.read())).open('SMSSpamCollection'), sep='\t', quoting=csv.QUOTE_NONE, names=("label", "message"))

    if name == 'Social_Network_Ads':
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/Social_Network_Ads/Social_Network_Ads.csv")), encoding='utf8', sep=",")
        print("Social Network Ads is a public dataset that can be used to determine what audience a car company should target in its ads in order to sell a SUV on a social network website.\n")
        return df
        #url = urllib.request.urlopen("https://github.com/daniel-yj-yang/machlearn/raw/master/machlearn/datasets/public/Social_Network_Ads/Social_Network_Ads.csv")
        #df = pd.read_csv(io.BytesIO(url.read()), encoding='utf8', sep=",")

    if name == 'Fashion_MNIST':
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


