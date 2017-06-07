#!/usr/bin/env python3

import sys
from operator import itemgetter

import numpy as np

from sklearn.feature_selection.text import TfidfVectorizer
from sklearn.metricx.pairwise import cosine_similarity
from sklearn.cluster

from olib import np, IOrganizer, interface


class lsiOrganizer(IOrganizer):
    pass


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
