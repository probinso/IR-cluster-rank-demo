#!/usr/bin/env python3

from functools import partial
import json
from operator import or_, itemgetter
import re
import sys

sys.path.append('./query') # out of time for clean code

import nltk
from nltk.stem.snowball import SnowballStemmer

from aug_lookup import AugmentedTFLookupTable
from term_frequency import TFDocument
from kmidf import CosKMIDFROrganizer
from ldaidf import LDAIDFROrganizer

from olib import augmented_interface as gobot
#from olib import interface as gobot

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

language  = 'english'
stopwords = set(nltk.corpus.stopwords.words(language))
stemmer   = SnowballStemmer(language)

def tokenize(contents):
    tokens = (word
              for sent in nltk.sent_tokenize(contents)
              for word in nltk.word_tokenize(sent))

    isnotstop = lambda s: s not in stopwords
    isword    = partial(re.search, '^[A-Za-z]*$')

    stems = map(stemmer.stem,
                filter(isword,
                       filter(isnotstop, tokens)))
    return [s for s in stems if s not in stopwords]

tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, max_features=200000,
    min_df=0.05, stop_words=language,
    use_idf=True, tokenizer=tokenize)


class JSONTFLookupTable(AugmentedTFLookupTable):
    def _extract(self, idx, struct_doc):
        doc = json.loads(struct_doc)
        doc_id   = doc['proj_title']
        contents = doc['abstract_text']
        return doc_id, contents

    def _transform(self, contents):
        return tokenize(contents)


###################################################


def interface(ifname, qstr, DOCID):

    # phase zero
    lookup = JSONTFLookupTable(TFDocument)
    lookup.populate(ifname)

    tdoc   = lookup.corpus.get(DOCID, None)
    if tdoc:
        print(tdoc['contents'])
        print(tdoc['title'])
    else:
        print(tdoc)

    _ = lookup.query(or_, *qstr.split())
    rank = sorted(_.items(), key=itemgetter(1), reverse=True)

    docs   = []
    titles = []
    target = None
    for place, payload in enumerate(rank):
        key, score = payload

        docs.append(lookup.corpus[key]['contents'])
        titles.append(lookup.corpus[key]['title'])

        if key == DOCID:
            target = place
            print("BEAT", place)

    if target is None:
        target = 0

    titles = np.array(titles)
    matrix = tfidf_vectorizer.fit_transform(docs)
    terms  = tfidf_vectorizer.get_feature_names()

    query  = np.where(terms==qstr.split())[0] # fuck

    #gobot(titles[target], matrix.toarray(), titles, terms,
    #      CosKMIDFROrganizer, query)
    gobot(titles[target],
          matrix.toarray(), titles, terms,
          CosKMIDFROrganizer, query)


def cli_interface():
    """
    by convention it is helpful to have a wrapper_cli method
    that interfaces from commandline to function space.
    """
    try:
        inpath  = sys.argv[1]
    except:
        print("usage: {}  <inpath> ".
              format(sys.argv[0]))
        sys.exit(1)
    #interface(inpath, 'vari unavail necessari', 802)
    interface(inpath, 'motor younger medic vivo', 9871) # 52
    #interface(inpath, 'molecular age disorder', 9871) 
    interface(inpath, 'motor age medic vivo', 9871) # 25


if __name__ == '__main__':
    cli_interface()
