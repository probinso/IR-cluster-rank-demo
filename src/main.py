#!/usr/bin/env python3

import csv
import json
import sys

from operator  import itemgetter
from functools import partial

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def select(D, rank):
    alg = KMeans(n_clusters=3)

    def represent(D):
        result  = alg.fit(D)
        labels  = result.labels_
        centers = result.cluster_centers_
        display  = dict()
        complete = dict()
        for l in np.unique(labels):
            keys = np.array([i for i, b in enumerate(labels==l) if b])
            ctr  = centers[l]

            gk = rank(D[keys, :])

            display[l]  = keys[gk]
            complete[l] = keys
        return display, complete

    DD = D
    while True:
        display, complete = represent(DD)
        yield display
        select = (yield)
        if select == -1:
            break
        DD = DD[complete[select], :]

    return complete[select]


def centroid_rank(n_display, M):
    ctr = np.mean(M, 0)
    gk  = [x[0] for x in
           sorted(enumerate(cdist(M, np.array([ctr]))),
                  key=itemgetter(1))[:n_display]]
    return gk


def tojson(groups):
    return json.dumps({str(g):groups[g].tolist() for g in groups})        
        

def interface(inpath):
    with open(inpath, 'rb') as fd:
        data = np.loadtxt(fd, delimiter=',', skiprows=1).astype('float')

    for result in process(data, tojson):
        print(result)


def process(data, operation):
    XY  = data[:, 0:2]

    crt   = select(XY, partial(centroid_rank, 3))
    groups = next(crt)
    while True:
        yield operation(groups)

        _   = next(crt)
        key = int(input('>> '))
        groups = crt.send(key)


def cli_interface():
    """
    by convention it is helpful to have a wrapper_cli method that interfaces
    from commandline to function space.
    """
    try:
        inpath  = sys.argv[1]
    except:
        print("usage: {}  <inpath>".format(sys.argv[0]))
        sys.exit(1)
    interface(inpath)


if __name__ == '__main__':
    cli_interface()
