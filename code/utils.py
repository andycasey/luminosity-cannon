#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" General utilities. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import sys


class ProgressBar():

    def __init__(self, header=None, show=True):
        self.header, self.show = header, show


    def __enter__(self):
        if self.show and self.header is not None:
            sys.stdout.write("\r{}\n".format(self.header))
        sys.stdout.flush()
        return self

    def __exit__(self, *args):
        if self.show:
            sys.stdout.write("\r\n")
            sys.stdout.flush()

    def update(self, i, N):
        """ Show the progress from i to N. """
        if not self.show: return 

        increment = max(1, int(N / 100))
        if i == 0 or i % increment == 0:
            sys.stdout.write("\r[{done}{not_done}] {percent:3.0f}%".format(
                done="=" * int((i + 1) / increment),
                not_done=" " * int((N - i - 1)/increment),
                percent=100. * (i + 1)/N))
            sys.stdout.flush()


