#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" General utilities. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import sys
import logging
from time import time

logger = logging.getLogger("cannon")

def progressbar(iterable, message=None, size=100):
    """
    A progressbar.

    :param iterable:
        Some iterable to show progress for.

    :type iterable:
        iterable

    :param message: [optional]
        A string message to show as the progressbar header.

    :type message:
        str

    :param size: [optional]
        The size of the progressbar. If the size given is zero or negative, then
        no progressbar will be shown.

    :type size:
        int
    """

    t_init = time()
    count = len(iterable)
    def _update(i, t=None):
        if 0 >= size: return
        increment = max(1, int(count / 100))
        if i % increment == 0 or i in (0, count):
            sys.stdout.write("\r[{done}{not_done}] {percent:3.0f}%{t}".format(
                done="=" * int(i/increment),
                not_done=" " * int((count - i)/increment),
                percent=100. * i/count,
                t="" if t is None else " ({0:.0f}s)".format(t-t_init)))
            sys.stdout.flush()

    # Initialise
    if size > 0:
        logger.info((message or "").rstrip())
        sys.stdout.flush()

    for i, item in enumerate(iterable):
        yield item
        _update(i)

    if size > 0:
        _update(count, time())
        sys.stdout.write("\r\n")
        sys.stdout.flush()


def label_vector(labels, order, cross_term_order=0):
    """
    Build a label vector description.
    """

    elements = []
    for label in labels:
        for i in range(order):
            _ = label if i == 0 else "{0}^{1:.0f}".format(label, i)
            elements.append(_)


    # For each label, do up to the order required.

    # If cross-terms are required, panic.


