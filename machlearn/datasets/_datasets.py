# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

# Some references:
# https://huggingface.co/docs/datasets/
# https://scikit-learn.org/stable/datasets/index.html#general-dataset-api

from io import BytesIO
from zipfile import ZipFile
import urllib.request
import csv

import pandas as pd

class dataset():
    """
    This class provides an interface to download dataset that is publicly available in the Internet.
    """

    def __init__(self, dataset_name=None):
        self.df = None
        self.dataset_name = dataset_name

    def download_as_df(self):
        """
        dataset_name can be the following:

            SMS_spam
        """
        if self.dataset_name is None:
            raise TypeError('dataset name is not provided')
        if self.dataset_name == 'SMS_spam':
            url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip")
            self.df = pd.read_csv(ZipFile(BytesIO(url.read())).open('SMSSpamCollection'), sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])
        return self.df

    

