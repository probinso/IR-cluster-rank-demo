#!/usr/bin/env python3

import sys
from operator import itemgetter

import numpy as np

import nltk
from nltk.stem.snowball import SnowballStemmer

import sklearn
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from olib import IOrganizer, interface


class cosidfOrganizer(IOrganizer):
    pass

def cli_interface():
    """
    by convention it is helpful to have a wrapper_cli method that interfaces
    from commandline to function space.
    """
    try:
        data_path = sys.argv[1]
        feat_path = sys.argv[2]
    except:
        print("usage: {}  <feature_path> <name_path>".format(sys.argv[0]))
        sys.exit(1)

    with open(data_path, 'rb') as fd:
        data = np.loadtxt(fd, delimiter=',', skiprows=1)

    with open(feat_path, 'rb') as fd:
        names = np.loadtxt(fd, delimiter=',', skiprows=1, dtype='|S100')

    interface(data, names, IOrganizer)


if __name__ == '__main__':
    cli_interface()
