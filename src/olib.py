#!/usr/bin/env python3

import json
import sys

import numpy as np


class Handler:
    def __init__(self, features, names, cls):
        """
          features : document vectors
          names    : document names
          cls      : Organizer
        """
        self.features = features.view(cls)
        self.names    = names

    def selector(self, clstr_count=3, show=3):
        D = self.features
        C = self.names

        while True:

            cl_idx = D.cluster(clstr_count)
            clstrs = {k: D[v] for k, v in cl_idx.items()}
            _names = {k: C[v] for k, v in cl_idx.items()}

            rk_idx = {k: v.rank() for k, v in clstrs.items()}
            ranked = {k: clstrs[k][v] for k, v in rk_idx.items()}
            _names = {k: _names[k][v] for k, v in rk_idx.items()}

            yield {k: _names[k][0:show] for k, v in ranked.items()}
            select = (yield)

            if select not in ranked:
                break
            D = ranked[select]
            C = _names[select]

        # XXX : Better terminal case
        return ranked


class Organizer(np.ndarray):
    """
      DATA : np.array where each index represents a document vector
    """
    def cluster(self, *args, **kwargs):
        raise NotImplemented

    def rank(self, *args, **kwargs):
        raise NotImplemented

    @property
    def _count(self):
        return np.size(self, 0)

    @property
    def _all_idx(self):
        return np.arange(0, self._count)


class IRelivance(Organizer):
    def relivance(self):
        """
          this returns the indicies of all documents
        """
        return self._all_idx


class ICluster(Organizer):
    def cluster(self, number=3):
        """
          Given a count of clusters, return indicies as a dictionary
          that indicate documents in each cluster
        """
        def split_padded(a,n):
            padding = (-len(a))%n
            return np.split(np.concatenate((a,np.zeros(padding))),n)

        clusters = split_padded(self._all_idx, number)
        return {k: v.astype('int') for k, v in enumerate(clusters)}


class IRank(Organizer):
    def rank(self):
        """
          Idenity rank sorts on index, then returns a subset of the indicies
        """
        return np.sort(self._all_idx)


class IOrganizer(IRank, ICluster, IRelivance):
    """
      Identity Organizer does no work for it's client
    """
    pass


def tojson(groups):
    trfm = lambda x : x.decode('utf-8')
    d = {str(k): [trfm(x) for x in v.tolist()] for k, v in groups.items()}
    return json.dumps(d)


def interface(data, names, cls):

    io = Handler(data, names, cls)

    ctr = io.selector()
    result = next(ctr)
    while ctr:
        print(tojson(result))
        _   = next(ctr)
        key = int(input('\n>> '))
        result = ctr.send(key)


def cli_interface(cls=IOrganizer):
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

    interface(data, names, cls)


if __name__ == '__main__':
    cli_interface()
