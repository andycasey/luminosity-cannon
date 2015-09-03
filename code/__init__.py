#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" The Cannon for absolute stellar luminosities. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import logging
import numpy as np
from warnings import simplefilter

# Speak up.
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cannon")

# Shut up.
simplefilter("ignore", np.RankWarning)
simplefilter("ignore", RuntimeWarning)