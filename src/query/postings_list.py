#!/usr/bin/env python3

# Batteries
from collections import namedtuple
from functools   import partial, reduce

# Installed Modules
from sortedcontainers import SortedList

# Local
from utilities import SortedDefaultDict


class _OpLoader:
    """
    We require all document formats to have an `doc_id` feild, for documents
    comparison operators which are only ids values
    __eq__
    __lt__
    """
    def __eq__(self, other):
        return self.doc_id == other.doc_id

    def __lt__(self, other):
        return self.doc_id < other.doc_id


class Document(_OpLoader, namedtuple("Document", ["doc_id"])):
    """
    TFD is a 'Term Frequency Document' that stores a doc_id and word_count
    """
    pass


class InvertedIndex(SortedDefaultDict):
    pass


class LookupTable(InvertedIndex):
    """
    InvertedIndex extended with PostingList as default constructor
    additionally 'query' object syntax for ops across entire table
    """
    def __init__(self, DocType, *args, **kwargs):
        assert(callable(DocType))

        self._DocType = DocType
        self._constructor = partial(PostingList, self._DocType)

        super().__init__(self._constructor, *args, **kwargs)

    def query(self, operator, *words):
        if len(words) == 0:
            return self._constructor()

        acc = sorted([self[w] for w in words], key=len)
        return reduce(operator, acc)


class PostingList(SortedList):
    """
    Extends SortedList with 'and' operator that performs an intersection with
    another PostingList
    """
    def __init__(self, constructor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(callable(constructor))
        self._constructor = constructor

    def __and__(self, other):
        """
        the exact INTERSECT algorithm from Manning Chapter 1
        """
        done = object()
        safe_next = lambda iterator: next(iterator, done)
        it  = iter(self)
        jt  = iter(other)

        acc = PostingList(self._constructor)

        i_doc, j_doc = safe_next(it), safe_next(jt)
        while True:
            if i_doc is done or j_doc is done:
                break

            if i_doc == j_doc:
                acc.append(i_doc)
                i_doc = safe_next(it)
                j_doc = safe_next(jt)
                continue

            if i_doc < j_doc:
                i_doc = safe_next(it)
            else:
                j_doc = safe_next(jt)

        return acc

    def __or__(self, other):
        return other

    def add(self, *args):
        super().add(self._constructor(*args))

