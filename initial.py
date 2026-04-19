# avoid repeating import sentences
# use `from imports import *` in your notebook
# make sure this file is in the path of import
# you may need to add the path using sys.path.append()

import logging
import math
import os
from copy import deepcopy
from itertools import combinations

import healpy as hp
import healsparse as hsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.table import Row, Table

# matplotlib settings
plt.rcParams["figure.dpi"] = 300
desired_font = "Source Serif 4"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [desired_font] + plt.rcParams["font.serif"]

# logging settings
logging.basicConfig(level=logging.INFO)

logging.info("Initialization complete.")
