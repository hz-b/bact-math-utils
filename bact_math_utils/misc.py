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
from typing import Iterator, Sequence


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


class CountSame:
    """Enumerate how long the same value is seen

    Follows :func:`enumerate` API

    Warning:
        Internally it uses '==' comparisons. So if using float values
        it can be good to scale the values and cast them to integers
        before using this class.

          value None will be considered as never seen
    """

    def __init__(self):
        self.last_item = None
        self.counter = None

    def __call__(self, iterable):
        for item in iterable:
            if item != self.last_item:
                self.counter = itertools.count()
                self.last_item = item

            assert self.counter is not None
            yield next(self.counter), item


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


def exhaust_and_add_if_required(seq: Sequence, *, prefix) -> Iterator:
    """Make a iterator of the given sequence, use prefix for filling up

    Warning:
        Unchecked function, use with care

    Args:
        seq: a sequence
        prefix: prefix to use for remaining arguments if required

    Similar and inspired to :func:`itertools.zip_longest`

    Yields item by item from the sequence. If further items are
    requested, the prefix is used to generate new items. It uses
    internally a counter. This counter is converted to the string
    and added to the prefix.

    Useful to be used to zip two sequences but contary to
    :func:`itertools.zip_longest`  a constant value is not
    sufficient. E.g. you need to add a label to each item in an
    other sequence
    """
    for s in seq:
        yield s
        counter = itertools.count()
        for cnt in counter:
            yield prefix + str(cnt)


__all__ = [
    "CountSame",
    "EnumerateUniqueEverSeen",
    "EnumerateUniqueJustSeen",
    "enumerate_changed_value",
    "enumerate_changed_value_tuple",
    "enumerate_changed_value_pairs",
]
