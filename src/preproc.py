#!/usr/bin/env python3

from collections import deque
from functools import partial
from json import loads
import re
import sys
import json

import sklearn
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem.snowball import SnowballStemmer

import numpy as np

from olib import IOrganizer, interface


language  = 'english'
stopwords = set(nltk.corpus.stopwords.words(language))
stemmer   = SnowballStemmer(language)

terms = None

def tokenize(text):
    tokens = (word
     for sent in nltk.sent_tokenize(text)
     for word in nltk.word_tokenize(sent))

    onlywords = partial(re.search, '^[A-Za-z]*$')

    stems = map(stemmer.stem, filter(onlywords, tokens))
    return [s for s in stems if s not in stopwords]


tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, max_features=200000,
    min_df=0.05, stop_words=language,
    use_idf=True, tokenizer=tokenize)


class CosKMOrganizer(IOrganizer):
    def cluster(self, n_clusters=3):
        dist = 1 - cosine_similarity(self)
        alg  = KMeans(n_clusters=n_clusters)

        # Fit cosine distance kmeans
        results = alg.fit(dist)
        labels  = results.labels_
        complete = dict()
        ulabels = np.unique(labels)
        for l in ulabels:
            keys = np.array([i for i, b in enumerate(labels==l) if b])
            complete[l] = keys

        # Centers used to identify human readable topics
        global terms
        centers  = results.cluster_centers_.argsort()[:, ::-1]
        metadata = None

        return complete , None


def process(data_path):
    with open(data_path) as fd:
        documents, titles = [], []
        for struct in (loads(line) for line in  fd):

            documents.append(struct['abstract_text'])
            titles.append(struct['proj_title'])

    titles = np.array(titles)
    matrix = tfidf_vectorizer.fit_transform(documents)
    terms  = np.array(tfidf_vectorizer.get_feature_names())

    return matrix.toarray(), titles, terms

def tojson(groups):
    trfm = lambda x : x.decode('utf-8') if not isinstance(x, str) else x

    d = {str(g): {topic : groups[g][topic].tolist() for topic in groups[g]} for g in groups}

    return json.dumps(d)


def cli_interface():
    """
    by convention it is helpful to have a wrapper_cli method that interfaces
    from commandline to function space.
    """
    try:
        data_path = sys.argv[1]
    except:
        print("usage: {}  <feature_path>".format(sys.argv[0]))
        sys.exit(1)

    global terms
    matrix, titles, terms = process(data_path)
    
    interface(matrix, titles, CosKMOrganizer)


if __name__ == '__main__':
    cli_interface()
