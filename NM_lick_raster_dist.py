"""
Plots: 
    - the distribution of the licks likelihood from Lightning Pose for NM sessions
    - distribution zoomed in to the middle ranges (>0, <0.9)
    - raster of lick events per trial, aligned to feedback time, for correct trials only

KB Mar2026
"""

#%%
""" imports ____________________________________________________________________________________________"""
import logging
import time
from pathlib import Path
import traceback
from string import ascii_uppercase
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.signal
from scipy.stats import ttest_ind, mannwhitneyu, ttest_1samp
import matplotlib.pyplot as plt
from ibllib.plots.snapshot import ReportSnapshotProbe, ReportSnapshot
from one.api import ONE
import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound
from ibllib.io.video import get_video_frame, url_from_eid
from brainbox.behavior.dlc import (
    likelihood_threshold, plt_window, get_speed, insert_idx,
    T_BIN, WINDOW_LEN, WINDOW_LAG, SAMPLING, _bin_window_licks)
from brainbox.behavior import training
from iblutil.numerical import ismember
from ibllib.plots.misc import Density
import json

WINDOW_LAG = -0.4
one = ONE()
THRESHOLD = 0.9
DIVIDER = 4

""" imports ____________________________________________________________________________________________"""
