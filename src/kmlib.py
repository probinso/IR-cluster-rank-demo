#!/usr/bin/env python3

import sys
from operator import itemgetter

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

from olib import IOrganizer, interface


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

    interface(data, names, kmOrganizer)


if __name__ == '__main__':
    cli_interface()
