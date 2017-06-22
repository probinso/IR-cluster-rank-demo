#!/usr/bin/env python3

# Batteries
from functools import partial
import json
from operator import and_, or_, itemgetter
import random
import re
import sys

# Local
from term_frequency import TFLookupTable, TFDocument, Counter

import nltk
from nltk.stem.snowball import SnowballStemmer

language  = 'english'
stopwords = set(nltk.corpus.stopwords.words(language))
stemmer   = SnowballStemmer(language)


class AugmentedTFLookupTable(TFLookupTable):
    """
    Extends TFLookuptable to load all corpus in memory as well
    """
    def __init__(self, *args, **kwargs):
        self.corpus = dict()
        super().__init__(*args, **kwargs)

    def _load_from(self, idx, struct_doc):
        doc_id, contents = self._extract(idx, struct_doc)

        self.corpus[idx] = {'title': doc_id, 'contents' : contents}

        tf = Counter(self._transform(contents))
        for term in tf:
            self[term].add(idx, tf[term])

    def _randdoc(self):
        return random.choice(list(self.corpus))

    def _distribution(self, idx):
        document = self.corpus[idx]

        contents = document['contents']
        tokens   = self._transform(contents)
        # print(tokens)

        dist = dict()
        for s in set(tokens):
            results = self.query(and_, s)
            dist[s] = results[idx]
        return dist


class JSONTFLookupTable(AugmentedTFLookupTable):
    def _extract(self, idx, struct_doc):
        doc = json.loads(struct_doc)
        doc_id   = doc['proj_title']
        contents = doc['abstract_text']
        return doc_id, contents

    def _transform(self, contents):
        tokens = (word
                  for sent in nltk.sent_tokenize(contents)
                  for word in nltk.word_tokenize(sent))

        isnotstop = lambda s: s not in stopwords
        isword    = partial(re.search, '^[A-Za-z]*$')

        stems = map(stemmer.stem, filter(isword, filter(isnotstop, tokens)))
        return [s for s in stems if s not in stopwords]


import numpy as np 


# phase one
def docselect(lookup):
    idx  = lookup._randdoc()
    dist = sorted(lookup._distribution(idx).items(), key=itemgetter(1), reverse=True)
    return idx, dist


# phase two
def qgenerator(dist, head=5, tail=-10, count=3):
    words, _ = zip(*dist[head:tail])
    scores   = _ / np.sum(_)

    qterms = np.random.choice(words, count, p=scores)
    return qterms


# phase three
def getresults(qterms, lookup):
    _ = lookup.query(or_, *qterms)
    results = sorted(_.items(), key=itemgetter(1), reverse=True)
    return results


# phase four
def display(qterms, results, lookup, idx):
    rank = next((i for i, v in enumerate(results) if v[0] == idx), -1)

    print(qterms)
    print(lookup.corpus[idx]['title'])

    print('located at ', rank, 'of', len(results))
    print()

    for i, pair in enumerate(results):
        key, value = pair
        print(lookup.corpus[key]['title'], value)
        if i > rank:
            break


def interface(ifname):

    # phase zero
    lookup = JSONTFLookupTable(TFDocument)
    lookup.populate(ifname)

    while True:

        cmd = input('>> ')
        if cmd == 'd':
            # phase one
            idx, dist = docselect(lookup)

        if cmd == 'q':
            head, tail, count = map(int, input('head, tail, count >>').split())
            # phase two
            head, tail, count = 5, -10, 3
            qterms = qgenerator(dist, head, tail, count)

        if cmd == 'r':
            # phase three
            results = getresults(qterms, lookup)

        if cmd == 'd':
            # phase four
            display(qterms, results, lookup, idx)

        if cmd == 's':
            name = input('SAVE PATH >>')

            # phase five
            ofname = 'jam_session.dat'
            with open(ofname) as fd:
                print(*qterms, file=fd)
                for key, _ in results:
                    print(key, file=fd)


def cli_interface():
    """
    by convention it is helpful to have a wrapper_cli method that interfaces
    from commandline to function space.
    """
    try:
        inpath  = sys.argv[1]
    except:
        print("usage: {}  <inpath>  <outpath>".format(sys.argv[0]))
        sys.exit(1)
    interface(inpath)


if __name__ == '__main__':
    cli_interface()
