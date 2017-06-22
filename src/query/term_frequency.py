#!/usr/bin/env python3

# Batteries
from collections import Counter, defaultdict, namedtuple

# Local
from postings_list import LookupTable


class TFDocument(namedtuple("TFDocument", ["doc_id", "count"])):
    """
    TFDocument is a 'Term Frequency Document' that stores a doc_id and word_count
    """
    pass


class TFLookupTable(LookupTable):
    """
    SortedDefaultDict extended with PostingList as default constructor
    additionally 'query' object syntax for ops across entire table
    """
    def populate(self, filename):
        with open(filename) as fd:
            for idx, struct_doc in enumerate(fd):
                self._load_from(idx, struct_doc)


    def _transform(self, contents):
        """
        transforms blob into tokens, provided as an overloaded function to
        be used for every non-readable operation, good place for stemmers
        """
        return contents.split()

    def _load_from(self, idx, struct_doc):
        doc_id, contents = self._extract(idx, struct_doc)

        tf = Counter(self._transform(contents))
        for term in tf:
            self[term].add(doc_id, tf[term])

    def query(self, operator, *words):
        twords = self._transform(' '.join(words))
        acc    = self._get_docs(operator, *twords)
        scored = self._score(acc, *twords)
        return scored

    def _get_docs(self, operator, *words):
        docs = super().query(operator, *words)
        acc  = LookupTable(self._constructor)

        for w in words:
            value = None
            for d in docs:
                doc_id, *_ = d

                try:
                    idx = self[w].index(d)
                except ValueError:
                    continue

                value = self[w][idx]

                try:
                    acc[w].append(value)
                except ValueError: # XXX
                    # Undefined behaviour, bad collision
                    # print(acc[w][-1])
                    # print(value)
                    continue

        return acc

    def _score(self, lookuptable, *words):
        dfreq = {w:len(self[w]) for w in words}
        divby = lambda den: lambda num: num/den

        acc = defaultdict(float)

        for w in dfreq:
            idf = divby(dfreq[w])
            for d in lookuptable[w]:
                doc_id, count = d
                acc[doc_id] += idf(count)
        return acc

