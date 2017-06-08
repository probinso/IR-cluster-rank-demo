#!/usr/bin/env python3

import sys
from operator import itemgetter

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

from olib import IOrganizer, interface, cli_interface


class kmOrganizer(IOrganizer):
    def cluster(self, n_clusters=3):
        alg = KMeans(n_clusters=n_clusters)

        results = alg.fit(self)
        labels  = results.labels_
        complete = dict()
        for l in np.unique(labels):
            keys = np.array([i for i, b in enumerate(labels==l) if b])
            complete[l] = keys
        return complete

    def rank(self):
        ctr = np.mean(self, 0)
        gk  = [x[0] for x in
               sorted(enumerate(cdist(self, np.array([ctr]))),
                      key=itemgetter(1))]
        return gk


if __name__ == '__main__':
    cli_interface(kmOrganizer)
