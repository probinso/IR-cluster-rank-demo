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


if __name__ == '__main__':
    cli_interface(cosidfOrganizer)
