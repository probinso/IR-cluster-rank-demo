#!/usr/bin/env Python3

from sortedcontainers import SortedDict, SortedList

class SortedDefaultDict(SortedDict):
    """
    extends the operation of DefaultDict class to SortedDict class
    """
    def __init__(self, constructor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(callable(constructor))
        self._constructor = constructor

    def __missing__(self, key):
        self[key] = self._constructor()
        return self[key]

