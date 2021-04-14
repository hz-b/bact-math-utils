'''Provide enumeration given that value changed receently  or completely new

Todo:
    find a new home for these tools

The naming follows
:func:`more_itertools.unique_everseen`
 and
func:`more_itertools.unique_lastseen`

Contrary to these, these class work use a string (not a single letter)
as an unique entry.
'''
import itertools
from collections import OrderedDict


class EnumerateUniqueEverSeen:
    '''Return unique id's for the item and the item

    Inspired by more_itertools.unique_everseen

    Stores the unique_items in :attr:`unique_items` as
    :class:`collection.OrderedDict`

    Follows :func:`enumerate`
    '''
    def __init__(self):
        self.unique_items = OrderedDict()
        self.counter = itertools.count()

    def __call__(self, iterable):
        for item in iterable:
            try:
                num = self.unique_items[item]
                yield num, item
                continue
            except KeyError:
                # Key does not exist ...
                pass
            cnt = next(self.counter)
            self.unique_items[item] = cnt
            yield cnt, item


class EnumerateUniqueJustSeen:
    '''Return unique id's for the item and the item

    Inspired by more_itertools.unique_lastseen
    Follows :func:`enumerate`
    '''
    def __init__(self):
        self.last_item = None
        self.cnt = None
        self.counter = itertools.count()

    def __call__(self, iterable):
        for item in iterable:
            if item == self.last_item:
                yield self.cnt, item
            else:
                self.cnt = next(self.counter)
                self.last_item = item
                yield self.cnt, item


__all__ = ['EnumerateUniqueEverSeen', 'EnumerateUniqueJustSeen']
