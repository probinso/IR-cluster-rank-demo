#!/usr/bin/env python3

from collections import deque
from functools import partial
from json import loads
from operator import itemgetter
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

from olib import Organizer, interface


language  = 'english'
stopwords = set(nltk.corpus.stopwords.words(language))
stemmer   = SnowballStemmer(language)


def tokenize(contents):
    tokens = (word
              for sent in nltk.sent_tokenize(contents)
              for word in nltk.word_tokenize(sent))

    isnotstop = lambda s: s not in stopwords
    isword    = partial(re.search, '^[A-Za-z]*$')

    stems = map(stemmer.stem, filter(isword, filter(isnotstop, tokens)))
    return [s for s in stems if s not in stopwords]


tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, max_features=200000,
    min_df=0.05, stop_words=language,
    use_idf=True, tokenizer=tokenize)


class IDFRankOrganizer(Organizer):
    def rank(self, queryvec):
        # provides cos(tfidf) ranking indicies against input query vector
        dist = cosine_similarity(self, queryvec)
        idx_dist = sorted(enumerate(dist), key=itemgetter(1), reverse=True)
        idx  = np.array([key for key, value in idx_dist])
        return idx


class CosKMOrganizer(Organizer):
    def cluster(self, terms, n_clusters=3):
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

        meta = dict()
        for l in complete:
            keys = complete[l]
            docs = self[keys, :]
            idf  = np.sum(docs, axis=0)
            sidx = np.array([key for key, value in
                             sorted(enumerate(idf), key=itemgetter(1))])
            meta[l] = [terms[idx] for idx in sidx[:6]]

        return complete, meta


class CosKMIDFROrganizer(CosKMOrganizer, IDFRankOrganizer):
    pass


def process(data_path):
    documents, titles = [], []
    with open(data_path) as fd:
        for struct in (loads(line) for line in  fd):
            documents.append(struct['abstract_text'])
            titles.append(struct['proj_title'])

    titles = np.array(titles)
    matrix = tfidf_vectorizer.fit_transform(documents)
    terms  = tfidf_vectorizer.get_feature_names()

    query  = tfidf_vectorizer.transform(['cancer'])

    return matrix.toarray(), titles, terms, query


def tojson(groups):

    trfm = lambda x : x.decode('utf-8') if not isinstance(x, str) else x
    d = {str(g):
         {'meta' : groups[g]['meta'], 'documents': groups[g]['documents'].tolist()}
         #{topic : groups[g][topic].tolist() for topic in groups[g]}
         for g in groups}

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

    #global terms
    
    matrix, titles, terms, query = process(data_path)

    interface(matrix, titles, terms, CosKMIDFROrganizer, query)


if __name__ == '__main__':
    cli_interface()
