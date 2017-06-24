
# Batteries
from operator import and_
import random

# Local
from term_frequency import TFLookupTable, Counter


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


