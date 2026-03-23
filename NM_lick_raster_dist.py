"""
Plots: 
    - the distribution of the licks likelihood from Lightning Pose for NM sessions
    - distribution zoomed in to the middle ranges (>0, <0.9)
    - raster of lick events per trial, aligned to feedback time, for correct trials only

Lick detection functions from here: 
    https://github.com/int-brain-lab/ibllib/blob/265f7dda7a40be554eedcccde7f686e1299019dd/brainbox/behavior/dlc.py#L127

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
from iblutil.numerical import bincount2D

one = ONE()
logger = logging.getLogger(__name__)  

WINDOW_LAG = -0.4
THRESHOLD = 0.9
DIVIDER = 4
cols_tongue = ['tongue_end_l_x', 'tongue_end_l_y',
          'tongue_end_r_x', 'tongue_end_r_y']


#%%
""" functions ____________________________________________________________________________________________"""
def valid_feature(x: str):
    if x.endswith('_x') or x.endswith('_y') or x.endswith('_likelihood'):
        return True
    return False



def likelihood_threshold(dlc, threshold=THRESHOLD):
    """Set dlc points with likelihood less than threshold to nan.

    :param dlc: dlc pqt object
    :param threshold: likelihood threshold
    :return:
    """
    features = np.unique(['_'.join(x.split('_')[:-1]) for x in dlc.keys() if valid_feature(x)])
    for feat in features:
        nan_fill = dlc[f'{feat}_likelihood'] < threshold
        dlc.loc[nan_fill, (f'{feat}_x', f'{feat}_y')] = np.nan
    return dlc


def get_feature_event_times(dlc, dlc_t, features, divider=DIVIDER):
    """
    Detect events from the dlc traces. Based on the standard deviation between frames
    :param dlc: dlc pqt table
    :param dlc_t: dlc times
    :param features: features to consider
    :return:
    """
    for i, feat in enumerate(features):
        f = dlc[feat]
        threshold = np.nanstd(np.diff(f)) / divider
        if i == 0:
            events = np.where(np.abs(np.diff(f)) > threshold)[0]
        else:
            events = np.r_[events, np.where(np.abs(np.diff(f)) > threshold)[0]]
    return dlc_t[np.unique(events)], threshold

def get_licks(dlc, dlc_t):
    """
    Compute lick times from the tongue dlc points
    :param dlc: dlc pqt table
    :param dlc_t: dlc times
    :return:
    """
    lick_times = get_feature_event_times(dlc, dlc_t, ['tongue_end_l_x', 'tongue_end_l_y',
                                                      'tongue_end_r_x', 'tongue_end_r_y'])
    return lick_times


def _bin_window_licks(lick_times, trials_df):
    """
    Helper function to bin and window the lick times and get them into trials df for plotting

    :param lick_times: np.array, timestamps of lick events
    :param trials_df: pd.DataFrame, with column 'feedback_times' (time of feedback for each trial)
    :returns: pd.DataFrame with binned, windowed lick times for plotting
    """
    # Bin the licks
    lick_bins, bin_times, _ = bincount2D(lick_times, np.ones(len(lick_times)), T_BIN)
    lick_bins = np.squeeze(lick_bins)
    start_window, end_window = plt_window(trials_df['feedback_times'])
    # Translating the time window into an index window
    try:
        start_idx = insert_idx(bin_times, start_window)
    except ValueError:
        logger.error('Lick time stamps are outside of the trials windows')
        raise
    end_idx = np.array(start_idx + int(WINDOW_LEN / T_BIN), dtype='int64')
    # Get the binned licks for each window
    trials_df['lick_bins'] = [lick_bins[start_idx[i]:end_idx[i]] for i in range(len(start_idx))]
    # Remove windows that the exceed bins
    trials_df['end_idx'] = end_idx
    trials_df = trials_df[trials_df['end_idx'] <= len(lick_bins)]
    return trials_df

def plot_lick_raster(lick_times, trials_df, ax):
    """
    Plots lick raster for correct trials

    :param lick_times: np.array, timestamps of lick events
    :param trials_df: pd.DataFrame, with column 'feedback_times' (time of feedback for each trial) and
                      feedbackType (1 for correct, -1 for incorrect trials)
    :returns: matplotlib.axis
    """
    licks_df = _bin_window_licks(lick_times, trials_df)
    correct = licks_df[licks_df['feedbackType'] == 1]['lick_bins']
    n_trials = len(correct)

    ax.imshow(list(correct), aspect='auto',
              extent=[-0.5, 1.5, n_trials, 0],
              cmap='gray_r',
              interpolation='none')
    ax.axvline(x=0, linestyle='--', color='purple', linewidth=1.5, label='feedback')
    ax.set_xticks([-0.5, 0, 0.5, 1, 1.5])
    ax.set_xlabel('time [sec]', fontsize=10)
    ax.set_ylabel('trials', fontsize=10)
    ax.set_title('Lick events per correct trial', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax
    # licks_df = _bin_window_licks(lick_times, trials_df)
    # plt.imshow(list(licks_df[licks_df['feedbackType'] == 1]['lick_bins']), aspect='auto',
    #            extent=[-0.5, 1.5, len(licks_df['lick_bins'].iloc[0]), 0], cmap='gray_r')
    # plt.xticks([-0.5, 0, 0.5, 1, 1.5])
    # plt.ylabel('trials')
    # plt.xlabel('time [sec]')
    # plt.axvline(x=0, label='feedback', linestyle='--', c='purple')
    # plt.title('Lick events per correct trial')
    # plt.tight_layout()
    # return plt.gca()

def plot_session(video_data_licks, video_data_licks_original, trials_df, session_name):
    # --- likelihood histograms ---
    edges = np.unique(np.round(np.concatenate([
        np.arange(0.00, 0.10 + 0.02, 0.02),
        np.arange(0.10, 0.90 + 0.02, 0.02),
        np.arange(0.90, 1.00 + 0.02, 0.02),
    ]), 6))

    l = video_data_licks["tongue_end_l_likelihood"].dropna()
    r = video_data_licks["tongue_end_r_likelihood"].dropna()

    counts_l, _ = np.histogram(l, bins=edges)
    counts_r, _ = np.histogram(r, bins=edges)
    # normalize to % of total frames
    total_l = len(l)
    total_r = len(r)
    density_l = counts_l / total_l * 100
    density_r = counts_r / total_r * 100

    # maximum detection on raw counts, ylim on density
    all_counts = np.maximum(counts_l, counts_r)
    max_bins = set()
    max_bins.add(np.argmax(all_counts))
    tmp = all_counts.copy(); tmp[list(max_bins)] = 0
    max_bins.add(np.argmax(tmp))
    remaining_density = np.maximum(density_l.copy(), density_r.copy())
    remaining_density[list(max_bins)] = 0
    ylim_subplot2 = remaining_density.max() * 1.15
    

    # --- lick events: threshold then detect ---
    video_data_licks_2 = likelihood_threshold(video_data_licks_original.copy(), threshold=THRESHOLD)
    lick_event_times, _ = get_feature_event_times(video_data_licks_2, video_data_licks_2['times'], cols_tongue, 2)
    lick_event_times = lick_event_times.to_frame(name='times')
    lick_times = lick_event_times['times'].to_numpy()

    # --- plot ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
    fig.suptitle(f"Session: {session_name}", fontsize=11, y=1.02)

    titles = ["Full scale", "Excluding highest bars around 0 and 1"]
    # ylims  = [None, ylim_subplot2]
    ylims = [None, 1.0]

    bin_centers = (edges[:-1] + edges[1:]) / 2

    for ax, ylim, title in zip(axes[:2], ylims, titles):
        ax.bar(bin_centers, density_l, width=np.diff(edges),
               color="#5bc0eb", alpha=0.8, label="Left",  align='center')
        ax.bar(bin_centers, density_r, width=np.diff(edges),
               color="#f7b538", alpha=0.8, label="Right", align='center')
        ax.set_xlim(0, 1)
        if ylim is not None:
            ax.set_ylim(0, ylim)
        for xv in [0.9]:
            ax.axvline(xv, color="black", linewidth=1, linestyle="--", alpha=0.75)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Likelihood", fontsize=10)
        ax.set_ylabel("% of frames", fontsize=10)       
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1f}%") 
        )
        ax.spines[["top", "right"]].set_visible(False)

    axes[1].legend(fontsize=9, frameon=False)

    # --- ax3: lick raster ---
    plot_lick_raster(lick_times, trials_df, axes[2])

    plt.tight_layout()
    safe_name = str(session_name).replace('/', '_')
    plt.show()
    # plt.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {safe_name}")

def plot_session_log(video_data_licks, video_data_licks_original, trials_df, session_name):
    edges = np.unique(np.round(np.concatenate([
        np.arange(0.00, 0.10 + 0.02, 0.02),
        np.arange(0.10, 0.90 + 0.02, 0.02),
        np.arange(0.90, 1.00 + 0.02, 0.02),
    ]), 6))

    l = video_data_licks["tongue_end_l_likelihood"].dropna()
    r = video_data_licks["tongue_end_r_likelihood"].dropna()

    counts_l, _ = np.histogram(l, bins=edges)
    counts_r, _ = np.histogram(r, bins=edges)
    total_l = len(l)
    total_r = len(r)
    density_l = counts_l / total_l
    density_r = counts_r / total_r

    # --- lick events ---
    video_data_licks_2 = likelihood_threshold(video_data_licks_original.copy(), threshold=THRESHOLD)
    lick_event_times, _ = get_feature_event_times(video_data_licks_2, video_data_licks_2['times'], cols_tongue, 2)
    lick_event_times = lick_event_times.to_frame(name='times')
    lick_times = lick_event_times['times'].to_numpy()

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    fig.suptitle(f"Session: {session_name}", fontsize=11, y=1.02)

    bin_centers = (edges[:-1] + edges[1:]) / 2

    axes[0].bar(bin_centers, density_l, width=np.diff(edges),
                color="#5bc0eb", alpha=0.8, label="Left", align='center')
    axes[0].bar(bin_centers, density_r, width=np.diff(edges),
                color="#f7b538", alpha=0.8, label="Right", align='center')
    axes[0].set_yscale('log')
    axes[0].set_xlim(0, 1)
    axes[0].axvline(0.9, color="black", linewidth=1, linestyle="--", alpha=0.75)
    axes[0].set_title("Full scale (log)", fontsize=10)
    axes[0].set_xlabel("Likelihood", fontsize=10)
    # axes[0].set_ylabel("% of frames", fontsize=10)
    # axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2g}%"))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2g}"))
    axes[0].set_ylabel("proportion of frames (log scale)", fontsize=10)    
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)

    # --- ax2: lick raster ---
    plot_lick_raster(lick_times, trials_df, axes[1])

    plt.tight_layout()
    safe_name = str(session_name).replace('/', '_')
    # plt.show()
    plt.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {safe_name}")


#%% 

save_dir = '/home/kceniabougrova/Documents/LP_licks_distribution/LP_LD_3_log'
os.makedirs(save_dir, exist_ok=True)

skipped_corrupted, skipped_no_tongue, skipped_other = [], [], []


#%% 

""" main loop ____________________________________________________________________________________________"""
# filter the eids with LP data
df_250 = pd.read_csv('/home/kceniabougrova/Documents/NM_project_fromIBLserver/NM_project2/data/LightningPoseSessions.xlsx - NoVideoSyncErrors.csv')
df_250 = df_250[df_250['LP'].notna()]
df_250_list = df_250['eid'].tolist()

for eid in df_250_list:
    try:
        df_lp = one.load_object(eid, 'leftCamera', attribute=['lightningPose', 'times'])
        video_data = pd.DataFrame(df_lp['lightningPose'])
        video_data["times"] = df_lp['times']

        cols = [
            'tongue_end_l_x', 'tongue_end_l_y', 'tongue_end_l_likelihood',
            'tongue_end_r_x', 'tongue_end_r_y', 'tongue_end_r_likelihood',
            'times'
        ]
        video_data_licks          = video_data[cols].copy()
        video_data_licks_original = video_data[cols].copy()

        trials = one.load_object(eid, 'trials')
        trials_extract = {k: v for k, v in trials.items()
                          if isinstance(v, np.ndarray) and v.ndim == 1}
        trials_df = pd.DataFrame(trials_extract)

        plot_session(video_data_licks, video_data_licks_original, trials_df, session_name=eid)

    except json.JSONDecodeError as e:
        print(f"[CORRUPTED] {eid}: {e}")
        skipped_corrupted.append(eid)
    except KeyError as e:
        print(f"[NO TONGUE COLS] {eid}: {e}")
        skipped_no_tongue.append(eid)
    except Exception as e:
        print(f"[OTHER] {eid}: {e}")
        skipped_other.append(eid)

print(f"\nDone. Corrupted: {len(skipped_corrupted)}, "
      f"No tongue: {len(skipped_no_tongue)}, "
      f"Other: {len(skipped_other)}")
# %%
""" main loop for log-scale____________________________________________________________________________________________"""
# filter the eids with LP data
df_250 = pd.read_csv('/home/kceniabougrova/Documents/NM_project_fromIBLserver/NM_project2/data/LightningPoseSessions.xlsx - NoVideoSyncErrors.csv')
df_250 = df_250[df_250['LP'].notna()]
df_250_list = df_250['eid'].tolist()

for eid in df_250_list:
    try:
        df_lp = one.load_object(eid, 'leftCamera', attribute=['lightningPose', 'times'])
        video_data = pd.DataFrame(df_lp['lightningPose'])
        video_data["times"] = df_lp['times']

        cols = [
            'tongue_end_l_x', 'tongue_end_l_y', 'tongue_end_l_likelihood',
            'tongue_end_r_x', 'tongue_end_r_y', 'tongue_end_r_likelihood',
            'times'
        ]
        video_data_licks          = video_data[cols].copy()
        video_data_licks_original = video_data[cols].copy()

        trials = one.load_object(eid, 'trials')
        trials_extract = {k: v for k, v in trials.items()
                          if isinstance(v, np.ndarray) and v.ndim == 1}
        trials_df = pd.DataFrame(trials_extract)

        plot_session_log(video_data_licks, video_data_licks_original, trials_df, session_name=eid)

    except json.JSONDecodeError as e:
        print(f"[CORRUPTED] {eid}: {e}")
        skipped_corrupted.append(eid)
    except KeyError as e:
        print(f"[NO TONGUE COLS] {eid}: {e}")
        skipped_no_tongue.append(eid)
    except Exception as e:
        print(f"[OTHER] {eid}: {e}")
        skipped_other.append(eid)

print(f"\nDone. Corrupted: {len(skipped_corrupted)}, "
      f"No tongue: {len(skipped_no_tongue)}, "
      f"Other: {len(skipped_other)}")
# %%
