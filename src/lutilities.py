import re
from functools import partial

import nltk
from nltk.stem.snowball import SnowballStemmer

language  = 'english'
stopwords = set(nltk.corpus.stopwords.words(language))
stemmer   = SnowballStemmer(language)

def tokenize(contents):
    tokens = (word
              for sent in nltk.sent_tokenize(contents)
              for word in nltk.word_tokenize(sent))

    isnotstop = lambda s: s not in stopwords
    isword    = partial(re.search, '^[A-Za-z]*$')

    stems = map(stemmer.stem, filter(isword,
                                     filter(isnotstop, tokens)))
    return [s for s in stems if s not in stopwords]


