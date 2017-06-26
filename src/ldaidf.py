#!/usr/bin/env python3

from operator import itemgetter
import sys

from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

from kmidf import IDFRankOrganizer, process
from olib  import Organizer, interface


class LDAOrganizer(Organizer):
    def cluster(self, terms, n_clusters=3):
        lda = LatentDirichletAllocation(n_topics=10*n_clusters)
        lda.fit(self)

        topics = lda.transform(self)

        # topic selection, using best LDA fit
        assigm = topics.argmax(axis=1)
        labels = np.unique(assigm)
        for i in range(10000):
            if labels.size >= n_clusters:
                break
            largest = max(labels, key=lambda l: np.sum(assigm==l))
            keys = np.where(assigm == largest)[0]
            topics[keys, largest] = np.power(topics[keys, largest], 2)

            assigm = topics.argmax(axis=1)
            labels = np.unique(assigm)

        # Insufficient diversity randomly splits
        while labels.size < n_clusters:
            largest = max(labels, key=lambda l: np.sum(assigm==l))
            keys = np.where(assigm == largest)[0]
            assigm[keys[:int(keys.size / 2)]] = max(labels) + 1
            labels = np.unique(assigm)

        # Too much diversity is merged
        while labels.size > n_clusters:
            smallest = sorted(labels, key=lambda l: np.sum(assigm==l))[:2]
            keys = np.where(assigm == smallest[0])
            assigm[keys] = smallest[1]
            labels = np.unique(assigm)

        complete = dict()
        for l in labels:
            complete[l] = np.where(assigm==l)[0]

        meta = dict()
        for l in complete:
            keys = complete[l]
            docs = self[keys, :]
            idf  = np.sum(docs, axis=0)
            sidx = np.array(
                [key for key, value in
                 sorted(enumerate(idf), key=itemgetter(1))])

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
