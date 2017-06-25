#!/usr/bin/env python3

from operator import itemgetter
import sys

from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

from kmidf import IDFRankOrganizer, process
from olib  import Organizer, interface
#from utilities import tokenize, language, stemmer


class LDAOrganizer(Organizer):
    def cluster(self, terms, n_clusters=3):
        lda = LatentDirichletAllocation(n_topics=n_clusters)
        lda.fit(self)

        topics = lda.transform(self)
        assigm = topics.argmax(axis=1)
        labels = np.unique(assigm)

        complete = dict()
        for l in labels:
            complete[l] = np.where(topics.argmax(axis=1)==l)

        meta = dict()
        for l in complete:
            keys = complete[l]
            docs = self[keys, :]
            idf  = np.sum(docs, axis=0)
            sidx = np.array([key for key, value in
                             sorted(enumerate(idf),
                                    key=itemgetter(1))])
            meta[l] = [terms[idx] for idx in sidx[:10]]

        return complete, meta


class LDAIDFROrganizer(LDAOrganizer, IDFRankOrganizer):
    pass


def cli_interface():
    """
    by convention it is helpful to have a wrapper_cli method 
    that interfaces from commandline to function space.
    """
    try:
        data_path = sys.argv[1]
    except:
        print("usage: {}  <feature_path>".format(sys.argv[0]))
        sys.exit(1)

    matrix, titles, terms, query = process(data_path)
    interface(matrix, titles, terms, LDAIDFROrganizer, query)


if __name__ == '__main__':
    cli_interface()
