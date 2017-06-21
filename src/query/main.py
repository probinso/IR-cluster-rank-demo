#!/usr/bin/env python3

# Batteries
import json
from operator import and_, itemgetter
import sys

# Local
from term_frequency import TFLookupTable, TFDocument

class CSVTFLookupTable(TFLookupTable):
    def _extract(self, idx, struct_doc):
        doc = json.loads(struct_doc)
        doc_id   = doc['proj_title']
        contents = doc['abstract_text']
        return doc_id, contents


def problem_3(ifname, *terms):
    print('\nProblem 3')
    lookup = CSVTFLookupTable(TFDocument)

    lookup.populate(ifname)
    results = lookup.query(and_, *terms)
    for doc in sorted(results.items(), key=itemgetter(1), reverse=True):
        print(*doc)


def interface(ifname):
    words = ['cancer']
    docs  = problem_3(ifname, *words)


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
