#!/usr/bin/env python3

import sys
from operator import itemgetter

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

from olib import IOrganizer, interface


class kmOrganizer(IOrganizer):
    def cluster(self):
        alg = KMeans(n_clusters=3)

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


def cli_interface():
    """
    by convention it is helpful to have a wrapper_cli method that interfaces
    from commandline to function space.
    """
    try:
        inpath = sys.argv[1]
    except:
        print("usage: {}  <inpath>".format(sys.argv[0]))
        sys.exit(1)
    interface(inpath, kmOrganizer)


if __name__ == '__main__':
    cli_interface()
