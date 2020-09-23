# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

# Some references:
# https://huggingface.co/docs/datasets/
# https://scikit-learn.org/stable/datasets/index.html#general-dataset-api

import io
from zipfile import ZipFile
import urllib.request
import csv

import pkgutil

import pandas as pd

def dataset(dataset_name=None):
    """
    dataset_name can be the following:

        - SMS_spam
        - Social_Network_Ads
    """
    if dataset_name is None:
        raise TypeError('dataset name is not provided')
    if dataset_name == 'SMS_spam':
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/SMS_spam_collection/SMSSpamCollection")), sep='\t', quoting=csv.QUOTE_NONE, names=("label", "message"))
        #url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip")
        #df = pd.read_csv(ZipFile(io.BytesIO(url.read())).open('SMSSpamCollection'), sep='\t', quoting=csv.QUOTE_NONE, names=("label", "message"))
    if dataset_name == 'Social_Network_Ads':
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/Social_Network_Ads/Social_Network_Ads.csv")), encoding='utf8', sep=",")
        #url = urllib.request.urlopen("https://github.com/daniel-yj-yang/machlearn/raw/master/machlearn/datasets/public/Social_Network_Ads.csv")
        #df = pd.read_csv(io.BytesIO(url.read()), encoding='utf8', sep=",")
    return df

    

