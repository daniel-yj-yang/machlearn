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

def public_dataset(name=None):
    """
    name can be one of the following:

        - SMS_spam
        - Social_Network_Ads

    Disclaimer:
    - The datasets collected are resources freely and publicly available in the Internet.
    - They are known as in the public domain.
    - They are stored in respective folders for the convenience of using this package.
    - The author does not own any of these datasets.
    """
    if name is None:
        raise TypeError('dataset name is not provided')
    if name == 'SMS_spam':
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/SMS_Spam_Collection/SMSSpamCollection.tsv")), sep='\t', quoting=csv.QUOTE_NONE, names=("label", "message"))
        #url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip")
        #df = pd.read_csv(ZipFile(io.BytesIO(url.read())).open('SMSSpamCollection'), sep='\t', quoting=csv.QUOTE_NONE, names=("label", "message"))
    if name == 'Social_Network_Ads':
        df = pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, "public/Social_Network_Ads/Social_Network_Ads.csv")), encoding='utf8', sep=",")
        #url = urllib.request.urlopen("https://github.com/daniel-yj-yang/machlearn/raw/master/machlearn/datasets/public/Social_Network_Ads/Social_Network_Ads.csv")
        #df = pd.read_csv(io.BytesIO(url.read()), encoding='utf8', sep=",")
    return df

    

