"""Provide enumeration given that value changed receently  or completely new

Todo:
    find a new home for these tools

The naming follows
:func:`more_itertools.unique_everseen`
 and
func:`more_itertools.unique_lastseen`

Contrary to these, these class work use a string (not a single letter)
as an unique entry.
"""
import itertools
from collections import OrderedDict
from typing import Sequence


class EnumerateUniqueEverSeen:
    """Return unique id's for the item and the item

    Inspired by :any:`more_itertools.unique_everseen`.

    Stores the unique_items in :attr:`unique_items` as
    :class:`collection.OrderedDict`

    Follows :func:`enumerate` API.

    Warning:
        Internally values are distinquished if :class:`collection.OrderedDict`
        sees them as different values.  So if using float values
        it can be good to scale the values and cast them to integers
        before using this class.

    """

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
    """Return unique id's for the item and the item

    Inspired by more_itertools.unique_lastseen.

    Follows :func:`enumerate` API

    Warning:
        Internally it uses '==' comparisons. So if using float values
        it can be good to scale the values and cast them to integers
        before using this class.
    """

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


def enumerate_changed_value(values: Sequence) -> list:
    """Emit a new number every time the value changes

    Uses :class:`EnumerateUniqueJustSeen`.

    Warning:
        see :class:`EnumerateUniqueJustSeen` when using float values.
    """

    mstep = EnumerateUniqueJustSeen()
    data = [cnt for cnt, val in mstep(values)]
    return data


def enumerate_changed_value_tuple(seq: Sequence) -> list:
    """Emit a new number every time one of the values in the sequence changes"""
    mstep = EnumerateUniqueJustSeen()
    data = [cnt for cnt, val in mstep(zip(*seq))]
    return data


def enumerate_changed_value_pairs(val1: Sequence, val2: Sequence) -> list:
    """Emit a new number every time one of the values changes

    Uses :class:`EnumerateUniqueJustSeen`.

    Warning:
        see :class:`EnumerateUniqueJustSeen` when using float values.
        Does not check both sequences are of equal length
    """
    return enumerate_changed_value_tuple((val1, val2))


__all__ = [
    "EnumerateUniqueEverSeen",
    "EnumerateUniqueJustSeen",
    "enumerate_changed_value",
    "enumerate_changed_value_tuple",
    "enumerate_changed_value_pairs",
]
