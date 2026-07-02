# avoid repeating import sentences
# use `from imports import *` in your notebook
# make sure this file is in the path of import
# you may need to add the path using sys.path.append()

import logging
import math
import multiprocessing
import os
from copy import deepcopy
from itertools import combinations

# Suppress NumExpr info logs (which print thread counts) before importing pandas
logging.getLogger("numexpr").setLevel(logging.WARNING)

import healpy as hp
import healsparse as hsp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.table import Row, Table
from astropy.wcs import WCS
from selasviz import launch_explorer

from hsc_wl.visual import *

# Use fork so worker processes (e.g. from dsigma.precompute) do not re-import
# the main module; this lets interactive scripts skip the
# `if __name__ == "__main__"` guard. No-op on Linux; skipped on Windows.
try:
    multiprocessing.set_start_method("fork", force=True)
except (ValueError, RuntimeError):
    pass

# matplotlib settings
plt.rcParams["figure.dpi"] = 300
desired_font = "Source Serif 4"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [desired_font] + plt.rcParams["font.serif"]

# logging settings
logging.basicConfig(level=logging.INFO)
