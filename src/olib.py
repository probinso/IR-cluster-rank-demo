#!/usr/bin/env python3

import json
import sys

import numpy as np


class Handler:
    def __init__(self, docmatrix, names, terms, cls, *rankargs):
        """
          docmatrix : document vectors
          names    : document names
          cls      : Organizer
          rankargs : rank parameters
        """
        self.docmatrix = docmatrix.view(cls) # cls
        self.names    = names
        self.terms    = terms
        self.rankargs = rankargs

    def selector(self, clstr_count=3, show=3):
        D = self.docmatrix
        C = self.names

        while True:

            cl_idx, meta = D.cluster(self.terms, clstr_count)

            clstrs = {k: D[v] for k, v in cl_idx.items()}
            _names = {k: C[v] for k, v in cl_idx.items()}

            rk_idx = {k: v.rank(*self.rankargs) for k, v in clstrs.items()}
            ranked = {k: clstrs[k][v] for k, v in rk_idx.items()}
            _names = {k: _names[k][v] for k, v in rk_idx.items()}

            yield {k: {'meta': meta[k], 'documents': _names[k][0:show]}
                   for k, v in ranked.items()}

            select = (yield)

            if select not in ranked:
                break
            D = ranked[select]
            C = _names[select]

        # XXX : Better terminal case
        return ranked


class AugmentedHandler(Handler):
    def __init__(self, target, docmatrix, names, terms, cls, *rankargs):
        super().__init__(docmatrix, names, terms, cls, *rankargs)
        self.target = target

    def selector(self, clstr_count=3, show=3):
        D = self.docmatrix
        C = self.names

        while True:

            cl_idx, meta = D.cluster(self.terms, clstr_count)

            clstrs = {k: D[v] for k, v in cl_idx.items()}
            _names = {k: C[v] for k, v in cl_idx.items()}

            rk_idx = {k: v.rank(*self.rankargs) for k, v in clstrs.items()}
            ranked = {k: clstrs[k][v] for k, v in rk_idx.items()}
            _names = {k: _names[k][v] for k, v in rk_idx.items()}

            for k in _names:
                if self.target in _names[k]:
                    meta[k].append('HERE!!!')

            yield {k: {'meta': meta[k], 'documents': _names[k][0:show]}
                   for k, v in ranked.items()}

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
    def cluster(self, terms, *args, **kwargs):
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
            padding = (-len(a)) % n
            return np.split(np.concatenate((a,np.zeros(padding))), n)

        clusters = split_padded(self._all_idx, number)
        return {k: v.astype('int') for k, v in enumerate(clusters)}, \
            {k: None for k, _ in enumerate(clusters)}


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
    trfm = lambda x : x.decode('utf-8') if not isinstance(x, str) else x
    d = {str(k): { l:
                   [trfm(x) for x in w.tolist()]
                   for l, w in v.items()}
         for k, v in groups.items()}

    return json.dumps(d)


import pprint
pp = pprint.PrettyPrinter(indent=4)

def augmented_interface(target, data, names, terms, cls, *rankargs):

    io = AugmentedHandler(target, data, names, terms, cls, *rankargs)

    ctr = io.selector()
    result = next(ctr)
    while ctr:
        pp.pprint(result)
        _   = next(ctr)
        key = int(input('\n>> '))
        result = ctr.send(key)


def interface(data, names, terms, cls, *rankargs):

    io = Handler(data, names, terms, cls, *rankargs)

    ctr = io.selector()
    result = next(ctr)
    while ctr:
        pp.pprint(result)
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
