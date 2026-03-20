#%%
"""
copy from Quadrado2025, but cleaned and with extra analyses after the labmeeting
"""
import numpy as np
import pandas as pd
import traceback
from pprint import pprint
from tqdm import tqdm
tqdm.pandas()
from datetime import datetime
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from one.api import ONE
# from brainbox.io.one import PhotometrySessionLoader

from iblphotometry import metrics
from iblphotometry.processing import z


#### SET PARAMETERS ############################################################

SESSIONS_FNAME = 'sessions_2025-11-07-12h07.pqt'
SESSION_TYPES = ['biased', 'ephys']
EIDS_TO_DROP = [
    'cd9d071e-c798-4900-891f-b65640ec22b1',  # huge photometry artifact (DR)
    '16aa7570-578f-4daa-8244-844716fb1320',  # huge photometry artifact (DR)
    'f4f1d7fe-d7c8-442b-a7d6-e214223febaf',  # huge photometry artifact (VTA)
    'a60531cd-e1e8-4b3b-b4d9-94b76ccc69c2',  # huge photometry artifact (VTA)
    '1c09046e-48d8-47f3-9d07-2241e3f3a136',  # huge photometry artifact (DR)
]
# '4ac35324-a13c-4517-a61f-7183a2f6ff44'  # severe movement artifacts (LC)
# '46fe69ff-d001-4608-a15e-d5e029c14fc3'  # extreme photobleaching (SNc)
# '69544b1b-7788-4b41-8cad-2d56d5958526'  # extreme photobleaching (SNc)
# '26e1b376-61dd-4d64-b0ab-ac4e6b8b9385'  # extreme photobleaching (SNc)
# '99d32415-3e41-468c-a21e-17f30063eb31'  # massive transients (VTA)
# '3cafedfc-b78b-48ba-9bce-0402b71bbe90'  # piece-wise signal (DR)
# n_unique samples >250 <500 don't seem terribly digitized, but mostly noise, not QC critical

EVENTS = ['stimOn_times', 'feedback_times']
N_TRIALS = 90
PSTH_WINDOW = (-1, 1)

RESPONSES_FNAME = 'responses_2025-11-10-15h29.pqt'
BASELINE_WINDOW = (-0.1, 0)
RESPONSE_WINDOW = (0.1, 0.35)

contrast_cmap = plt.get_cmap("inferno_r", 5)
CONTRAST_COLORS = {
    'contrast_0.0': contrast_cmap(0),
    'contrast_0.0625': contrast_cmap(1),
    'contrast_0.125': contrast_cmap(2),
    'contrast_0.25': contrast_cmap(3),
    'contrast_1.0': contrast_cmap(4),
}
NM_COLORS = {
    'DA':  '#de2d26',   # red gradient
    '5HT': '#8e44ad',   # purple gradient
    'NE':  '#2171b5',   # blue gradient
    'ACh': '#31a354'    # green gradient
}

NM_CMAPS = {
    'DA': plt.colormaps['Reds'],
    '5HT': plt.colormaps['Purples'],
    'NE': plt.colormaps['Blues'],
    'ACh': plt.colormaps['Greens'],
}

# Set font sizes (big for poster)
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})


#### DEFINE HELPER FUNCTIONS ###################################################
def get_responses(photometry, trials, event, time_window=PSTH_WINDOW):
    """Return peri-event aligned zdFF and time axis."""
    t = photometry.index.values
    SAMPLING_RATE = int(1 / np.mean(np.diff(t)))
    calcium = photometry.values
    t_events = trials[event].dropna().values
    t_events= t_events[
        (t_events + time_window[0] >= t.min()) & (t_events + time_window[1] <= t.max())
        ]
    n_trials = len(t_events)
    samples_window = np.arange(time_window[0]*SAMPLING_RATE, time_window[1]*SAMPLING_RATE)
    psth_idx = np.tile(samples_window[:, None], (1, n_trials))
    event_idx = np.searchsorted(t, t_events)
    psth_idx += event_idx
    # ~psth_idx = psth_idx[(psth_idx >= 0) & (psth_idx < len(t))].reshape(-1, n_trials)
    responses = calcium[psth_idx]
    return responses

def get_response_tpts(photometry, time_window=PSTH_WINDOW):
    t = photometry.index.values
    SAMPLING_RATE = int(1 / np.mean(np.diff(t)))
    samples_window = np.arange(time_window[0]*SAMPLING_RATE, time_window[1]*SAMPLING_RATE)
    return np.linspace(time_window[0], time_window[1], samples_window.shape[0])

def normalize_response(trial, bwin=(-0.1, 0), divide=True):
    i0, i1 = trial['tpts'].searchsorted(bwin)
    bval = trial['response'][i0:i1].mean()
    resp_norm = trial['response'] - bval
    if divide:
        resp_norm = resp_norm / bval
    return resp_norm

def resample_response(trial, new_tpts, fill_value=np.nan):
    """
    Resample response data to a new time base using linear interpolation.

    Parameters:
    -----------
    row : pd.Series
        Row containing 'tpts' and 'response' arrays
    new_timebase : np.ndarray
        Target time points for resampling
    fill_value : float, optional
        Value to use for points outside the original time range.
        Default is np.nan. Can also use a tuple (left_fill, right_fill)
        for different values on each side.

    Returns:
    --------
    np.ndarray
        Resampled response values at new_timebase points
    """
    return np.interp(
        new_tpts, trial['tpts'], trial['response'],
        left=fill_value, right=fill_value
        )

def get_response_magnitude(trial, method='mean', twindow=RESPONSE_WINDOW):
    i0, i1 = trial['tpts'].searchsorted(twindow)
    if i0 == i1:
        return np.nan
    if method == 'mean':
        return trial['response'][i0:i1].mean()
    elif method == 'slope':
        t = trial['tpts'][i0:i1]
        y = trial['response'][i0:i1]
        if len(y) < 5:
            return np.nan
        slope, _, _, _, _ = stats.linregress(t, y)
        return slope
    else:
        raise NotImplementedError

def plot_mean_response(
    trials, color='black', twindow=PSTH_WINDOW, plot_all=False, ax=None, **kwargs
    ):
    if ax is None:
        fig, ax = plt.subplots()

    if plot_all:
        for _, trial in trials.iterrows():
            ax.plot(trial['tpts'], trial['response'], color=color, alpha=0.1)

    tpts = trials['tpts'].iloc[0]
    responses = np.stack(trials['response'])
    mean = np.mean(responses, axis=0)
    sem = stats.sem(responses, axis=0)

    i0, i1 = tpts.searchsorted(PSTH_WINDOW)
    ax.plot(tpts[i0:i1], mean[i0:i1], color=color, **kwargs)
    ax.fill_between(
        tpts[i0:i1], (mean - sem)[i0:i1], (mean + sem)[i0:i1], alpha=0.25, color=color
        )

    return ax

def pval2stars(p, ns='n.s.', na='n/a'):
    if np.isnan(p):
        return na
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ns

def cm2in(cm):
    return cm / 2.54

def set_plotsize(w, h=None, ax=None):
    """
    Set the size of a matplotlib axes object in cm.

    Parameters
    ----------
    w, h : float
        Desired width and height of plot, if height is None, the axis will be
        square.

    ax : matplotlib.axes
        Axes to resize, if None the output of plt.gca() will be re-sized.

    Notes
    -----
    - Use after subplots_adjust (if adjustment is needed)
    - Matplotlib axis size is determined by the figure size and the subplot
      margins (r, l; given as a fraction of the figure size), i.e.
      w_ax = w_fig * (r - l)
    """
    if h is None: # assume square
        h = w
    w = cm2in(w) # convert cm to inches
    h = cm2in(h)
    if not ax: # get current axes
        ax = plt.gca()
    # get margins
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    # set fig dimensions to produce desired ax dimensions
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def nice_ticks(ymin, ymax, d=2, n_ticks=5):
    # Round bounds
    ymin = np.floor(ymin * 10**d) / 10**d
    ymax = np.ceil(ymax * 10**d) / 10**d

    if ymin < 0 < ymax:
        # Include 0 and space evenly on both sides
        spacing = (ymax - ymin) / n_ticks
        # ~spacing = np.round(spacing, d)

        # Ensure spacing is not zero after rounding
        if spacing == 0:
            spacing = 10**(-d)

        # Adjust bounds to align with step that includes 0
        ymin_adj = -np.ceil(abs(ymin) / spacing) * spacing
        ymax_adj = np.ceil(ymax / spacing) * spacing

        ticks = np.arange(ymin_adj, ymax_adj + spacing, spacing)
    else:
        ticks = np.linspace(ymin, ymax, n_ticks)

    return ticks

def clip_axes_to_ticks(ax=None, spines=['left', 'bottom'], ext={}):
    """
    Clip the axis lines to end at the minimum and maximum tick values.

    Parameters
    ----------
    ax : matplotlib.axes
        Axes to resize, if None the output of plt.gca() will be re-sized.

    spines : list
        Axes to keep and clip, axes not included in this list will be removed.
        Valid values include 'left', 'bottom', 'right', 'top'.

    ext : dict
        For each axis in ext.keys() ('left', 'bottom', 'right', 'top'),
        the axis line will be extended beyond the last tick by the value
        specified, e.g. {'left':[0.1, 0.2]} will results in an axis line
        that extends 0.1 units beyond the bottom tick and 0.2 unit beyond
        the top tick.
    """
    if ax is None:
        ax = plt.gca()
    spines2ax = {
        'left': ax.yaxis,
        'top': ax.xaxis,
        'right': ax.yaxis,
        'bottom': ax.xaxis
    }
    all_spines = ['left', 'bottom', 'right', 'top']
    for spine in spines:
        low = min(spines2ax[spine].get_majorticklocs())
        high = max(spines2ax[spine].get_majorticklocs())
        if spine in ext.keys():
            low += ext[spine][0]
            high += ext[spine][1]
        ax.spines[spine].set_bounds(low, high)
    for spine in [spine for spine in all_spines if spine not in spines]:
        ax.spines[spine].set_visible(False)


#%% 
"""
#### PREPARE DATA FOR ANALYSIS #################################################
"""

# Load the dataframe (in case you already ran the loop)
df_responses = pd.read_parquet('/home/kceniabougrova/Documents/NM_project_fromIBLserver/NM_project2/CQ_PosterSfN_DC/responses_2025-11-10-15h29.pqt')

# Drop trials where there was no response
df_responses = df_responses.query('choice != 0').copy()

# Drop trials where the reaction time is implausible
df_responses = df_responses.query('reaction_time > 0.05').copy()

# Drop ECW for now (some wierd photometry there)
df_responses = df_responses.query('session_type == "biased"')

# Print some metadata
n_mice = df_responses.groupby(['target', 'NM']).apply(
    lambda x: x['subject'].nunique(), include_groups=False
    )
print("N mice per target-NM")
print(n_mice)
n_sessions = df_responses.groupby(['target', 'NM']).apply(
    lambda x: x['eid'].nunique(), include_groups=False
    )
print("\nN sessions per target-NM")
print(n_sessions)

# Add convenience columns for analyses
df_responses = df_responses.dropna(subset='response')
df_responses['contrast'] = df_responses['signed_contrast'].apply(np.abs)
df_responses['hemisphere'] = df_responses['hemisphere'].apply(
    lambda x: 1 if x == 'r' else -1
    )
df_responses['relative_contrast'] = df_responses.apply(
    lambda x: x['signed_contrast'] * x['hemisphere'],
    axis='columns'
    )
df_responses['side'] = df_responses.apply(  # True is contra , False is ipsi
    lambda x: np.signbit(x['relative_contrast']), axis='columns'
    )

# ==================================================================
# BUGGED PROBABILITY LEFT SESSIONS 
# ------------------------------------------------------------------
import numpy as np
import pandas as pd

buggy_sessions = []
bug_info = []
MIN_BLOCK_LEN = 5 

for eid, df_sess in df_responses.groupby('eid'):
    p = df_sess['p_left'].values
    if len(p) < 5:
        continue

    # find where p_left changes
    change_idxs = np.where(p[1:] != p[:-1])[0] + 1
    block_lengths = np.diff(np.r_[0, change_idxs, len(p)])

    # skip sessions with only unbiased (0.5) or one biased block
    if len(change_idxs) < 2:
        continue

    # compute distances between consecutive changes
    diffs = np.diff(change_idxs)

    # A bug = two or more consecutive switches separated by < MIN_BLOCK_LEN
    short_flips = np.where(diffs < MIN_BLOCK_LEN)[0]

    if len(short_flips) > 0:
        buggy_sessions.append(eid)
        bug_info.append({
            "eid": eid,
            "change_idxs": change_idxs.tolist(),
            "diffs": diffs.tolist(),
            "short_flips_indices": change_idxs[short_flips].tolist(),
        })

print(f"⚠️ Found {len(buggy_sessions)} potentially buggy sessions:")
print(buggy_sessions)

df_bugs = pd.DataFrame(bug_info)


# remove all rows from df_responses whose eid is in buggy_sessions
bugs_list = df_bugs.eid
df_responses_clean = df_responses[~df_responses['eid'].isin(bugs_list)].copy()

print(f"Removed {df_responses.shape[0] - df_responses_clean.shape[0]} trials "
      f"from {len(bugs_list)} buggy sessions.")


df_responses = df_responses_clean.copy()


# ==================================================================
# Normalize the responses
# ------------------------------------------------------------------
df_responses.loc[:, 'response'] = df_responses.apply(
    normalize_response, axis='columns'
    )

# Resample the responses to a common time-base
new_tpts = np.linspace(-0.9, 1.9, 90)
df_responses.loc[:, 'response'] = df_responses.apply(
    lambda x: resample_response(x, new_tpts), axis='columns'
    )
df_responses.loc[:, 'tpts'] = df_responses.apply(lambda x: new_tpts, axis='columns')

# Get repsonse magnitudes
df_responses['response_mean'] = df_responses.progress_apply(
    lambda x: get_response_magnitude(x, method='mean', twindow=RESPONSE_WINDOW),
    # ~lambda x: get_response_magnitude(x, twindow=(0, x['firstMovement_times'])),
    axis='columns'
    )
# ~df_responses['response_slope'] = df_responses.progress_apply(
    # ~lambda x: get_response_magnitude(x, method='slope', twindow=RESPONSE_WINDOW),
    # ~axis='columns'
    # ~) 

df_original = df_responses.copy()
















#%% 

"""


################################################################################

################################################################################

###### PART 1 - RTs ############################################################

################################################################################

################################################################################


"""
#%%
"""
#### PLOT RESULTS ##############################################################
"""
df_responses = df_original[df_original.event == 'stimOn_times'].copy()
# df_responses = df_original[df_original.event == 'feedback_times'].copy()

# Plot log reaction time distributions for each contrast level
rts = [
    np.log10(t['reaction_time'].dropna().values)
    for _, t in df_responses.groupby('contrast')
    ]
fig, ax = plt.subplots()
parts = ax.violinplot(
    rts,
    showextrema=False,
    showmedians=True,
    # orientation='horizontal' #giving error "TypeError: Axes.violinplot() got an unexpected keyword argument 'orientation' 10122025" 
    vert=False
    )
cmap = plt.colormaps['Greys']
rt_colors = cmap(np.linspace(0.4, 0.99, len(rts)))
for pc, color in zip(parts['bodies'], rt_colors):
    pc.set_facecolor(color)
parts['cmedians'].set_colors(rt_colors)
for i, rt in enumerate(rts):
    median = np.median(rt)
    ax.text(
        median, i + 1, f'{10**median:.2f}s', rotation=45, ha='left', va='bottom', size=14
        )
contrasts = df_responses['contrast'].unique()
ax.set_yticks(np.arange(1, len(contrasts) + 1))
ax.set_yticklabels([f'{c*100:.0f}' for c in sorted(contrasts)])
ax.set_ylabel('Contrast level')
xticks = np.linspace(-2, 2, 6)
ax.set_xticks(xticks)
ax.set_xticklabels(['$10^{%d}$' % t for t in xticks])
ax.set_xlim([-1.5, 1.5])
ax.set_title(
    "Reaction Time Distributions by Contrast Level",
    fontsize=16,
    fontweight="bold",
    pad=31     # space between title and plot
)
ax.set_xlabel('Reaction time (s)')
clip_axes_to_ticks(ax=ax)
set_plotsize(w=10, h=5, ax=ax)
# fig.savefig('figures/reaction_times.svg')


#%%
"""
#############################################################################
trying RT only in the DR df
* plots split by correct/incorrect trials
"""
# df_original = df_responses #to save so I dont need to rerun ttthe code above
df_5HT = df_responses[df_responses.target == "DR"]
df_responses = df_5HT.copy()

# --- Split dataframe ---
df_corr = df_responses[df_responses.feedback == 1]
df_inc  = df_responses[df_responses.feedback == -1]

# --- Prepare contrasts ---
contrasts_sorted = sorted(df_responses['contrast'].unique())

# --- Create figure with 2 subplots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

datasets = [
    (df_corr, axes[0], "Correct trials"),
    (df_inc,  axes[1], "Incorrect trials")]

for df, ax, title in datasets:
    # --- RT distributions per contrast ---
    rts = [
        np.log10(t['reaction_time'].dropna().values)
        for _, t in df.groupby('contrast') ]

    # --- Violin plot ---
    parts = ax.violinplot(
        rts,
        showextrema=False,
        showmedians=True,
        # orientation='horizontal' #giving error "TypeError: Axes.violinplot() got an unexpected keyword argument 'orientation' 10122025" 
        vert=False)

    # --- Color map ---
    cmap = plt.colormaps['Greys']
    rt_colors = cmap(np.linspace(0.4, 0.99, len(rts)))

    for pc, color in zip(parts['bodies'], rt_colors):
        pc.set_facecolor(color)
    parts['cmedians'].set_colors(rt_colors)

    # --- Add median text ---
    for i, rt in enumerate(rts):
        if len(rt) == 0:
            continue
        median = np.median(rt)
        ax.text(
            median, i + 1, 
            f'{10**median:.2f}s',
            rotation=45,
            ha='left', va='bottom', size=14)

    # --- Y-axis labels ---
    ax.set_yticks(np.arange(1, len(contrasts_sorted) + 1))
    ax.set_yticklabels([f'{c*100:.0f}' for c in contrasts_sorted])
    ax.set_ylabel('Contrast level')

    # --- X-axis ticks ---
    xticks = np.linspace(-2, 2, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'$10^{int(t)}$' for t in xticks])
    # ax.set_xlim([-2.1, 2])
    ax.set_xlabel('Reaction time (s)')

    # --- Title ---
    ax.set_title(title)
    ax.set_title(
        title, 
        fontsize=16,
        # fontweight="bold",
        pad=31     # space between title and plot
    )
    clip_axes_to_ticks(ax=ax)

plt.tight_layout()
set_plotsize(w=25, h=8) 

#%%
"""
#########################################################################
separate by probability blocks
"""
# Split dataframe
df_corr = df_responses[df_responses.feedback == 1]
df_inc  = df_responses[df_responses.feedback == -1]

# Unique and sorted p_left values
p_left_values = sorted(df_responses.p_left.unique())   # [0.2, 0.5, 0.8]

# Prepare contrasts
contrasts_sorted = sorted(df_responses['contrast'].unique())

# Create figure with 2 rows × 3 columns
fig, axes = plt.subplots(
    2, 3,
    figsize=(22, 8),
    sharey=True,
    sharex=True, 
    gridspec_kw={'hspace': 0.55}   # increase vertical spacing 
)

datasets = [
    (df_corr, 0, "Correct trials"),    # first row
    (df_inc,  1, "Incorrect trials"),  # second row
]


for df_main, row_idx, row_title in datasets:

    for col_idx, p_left in enumerate(p_left_values):

        ax = axes[row_idx, col_idx]

        # Filter by p_left
        df = df_main[df_main.p_left == p_left]

        # RT distributions per contrast
        rts = [
            np.log10(t['reaction_time'].dropna().values)
            for _, t in df.groupby('contrast')
        ]

        # Violin plot
        parts = ax.violinplot(
            rts,
            showextrema=False,
            showmedians=True,
            vert=False
        )

        # Colors
        cmap = plt.colormaps['Greys']
        rt_colors = cmap(np.linspace(0.4, 0.99, len(rts)))

        for pc, color in zip(parts['bodies'], rt_colors):
            pc.set_facecolor(color)
        parts['cmedians'].set_colors(rt_colors)

        # Add medians
        for i, rt in enumerate(rts):
            if len(rt) == 0:
                continue
            median = np.median(rt)
            ax.text(
                median, i + 1,
                f'{10**median:.2f}s',
                rotation=45,
                ha='left', va='bottom',
                size=12
            )

        # -------------------------------
        # Y-axis only on the first column
        # -------------------------------
        ax.set_yticks(np.arange(1, len(contrasts_sorted) + 1))
        ax.set_yticklabels([f'{c*100:.0f}' for c in contrasts_sorted])

        if col_idx == 0:
            # First column → keep the axis label
            ax.set_ylabel('Contrast')
        else:
            # Other columns → keep tick values, remove axis label
            ax.set_ylabel('')


        # -------------------------------
        # X-axis only on the second row
        # -------------------------------
        xticks = np.linspace(-2, 2, 6)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'$10^{int(t)}$' for t in xticks])

        if row_idx == 0:
            # Remove x-axis labels/ticks for the first row
            ax.set_xlabel('')
            ax.set_xticklabels([])

        else:
            # Second row → show x-axis label
            ax.set_xlabel('Reaction time (s)')

        # Title for each subplot
        ax.set_title(
            f'{row_title} | p_left = {p_left}',
            fontsize=14,
            pad=25
        )

        clip_axes_to_ticks(ax=ax)


plt.tight_layout()
set_plotsize(w=28, h=10)


#%%
"""
#########################################################################
only DR
only unbiased block
RT split by correct/incorrect and choice side left/right 
""" 
import matplotlib.patches as mpatches

df_unbiased = df_responses[df_responses.p_left == 0.5]

df_corr = df_unbiased[df_unbiased.feedback == 1]
df_inc  = df_unbiased[df_unbiased.feedback == -1]

contrasts_sorted = sorted(df_unbiased['contrast'].unique())

color_left  = "#F9A65A"   # pastel orange (based on #F67B00)
color_right = "#69BDBF"   # pastel teal   (based on #0A9396)

fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 5),
    sharey=True,
    gridspec_kw={'wspace': 0.15})

datasets = [
    (df_corr, axes[0], "Correct trials"),
    (df_inc,  axes[1], "Incorrect trials")]

for df_main, ax, title in datasets:
    # Build RT arrays for each contrast AND each choice
    rts_left = []
    rts_right = []

    for c in contrasts_sorted:
        df_c = df_main[df_main.contrast == c]

        rt_left  = np.log10(df_c[df_c.choice == -1]['reaction_time'].dropna().values)
        rt_right = np.log10(df_c[df_c.choice == 1 ]['reaction_time'].dropna().values)

        rts_left.append(rt_left)
        rts_right.append(rt_right)

    # ---------------------------------------------------------
    # Plot violins side by side for left/right choices
    # ---------------------------------------------------------

    # Shift positions for left/right violins:
    pos = np.arange(1, len(contrasts_sorted) + 1)
    pos_left  = pos - 0.18
    pos_right = pos + 0.18

    # Left choice violins
    vp_left = ax.violinplot(
        rts_left,
        positions=pos_left,
        showextrema=False,
        showmedians=True,
        vert=False
    )
    for b in vp_left['bodies']:
        b.set_facecolor(color_left)
        b.set_alpha(0.65)
    vp_left['cmedians'].set_color(color_left)

    # Right choice violins
    vp_right = ax.violinplot(
        rts_right,
        positions=pos_right,
        showextrema=False,
        showmedians=True,
        vert=False
    )
    for b in vp_right['bodies']:
        b.set_facecolor(color_right)
        b.set_alpha(0.65)
    vp_right['cmedians'].set_color(color_right)

    # ---------------------------------------------------------
    # Add median text for both choices
    # ---------------------------------------------------------
    for i, (rtL, rtR) in enumerate(zip(rts_left, rts_right)):
        if len(rtL) > 0:
            mL = np.median(rtL)
            ax.text(mL, pos_left[i], f'{10**mL:.2f}s',
                    ha='left', va='bottom', rotation=45, size=15)
        if len(rtR) > 0:
            mR = np.median(rtR)
            ax.text(mR, pos_right[i], f'{10**mR:.2f}s',
                    ha='left', va='bottom', rotation=45, size=15)

    # ---------------------------------------------------------
    # Axes 
    # ---------------------------------------------------------
    ax.set_yticks(pos)
    ax.set_yticklabels([f'{c*100:.0f}' for c in contrasts_sorted])

    if col_idx == 0:
        # First column → keep the axis label
        ax.set_ylabel('Contrast')
    else:
        # Other columns → keep tick values, remove axis label
        ax.set_ylabel('')

    xticks = np.linspace(-2, 2, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'$10^{int(t)}$' for t in xticks])
    ax.set_xlabel("Reaction time (s)")

    ax.set_title(title, fontsize=16, pad=20)

    patch_left  = mpatches.Patch(color=color_left,  label='Left choice')
    patch_right = mpatches.Patch(color=color_right, label='Right choice')
    fig.legend(
        handles=[patch_left, patch_right],
        loc='center right',
        bbox_to_anchor=(1.04, 0.5),   # move legend outside the figure
        frameon=False,
        fontsize=14)
        
    clip_axes_to_ticks(ax=ax)

plt.tight_layout()
set_plotsize(w=20, h=10)



#%%
"""
##########################################################################
same but split by the probabilityLeft blocks 
2x3 grid
""" 
# --- Sort p_left values ---
p_left_values = sorted(df_responses.p_left.unique())   # [0.2, 0.5, 0.8]

df_corr_all = df_responses[df_responses.feedback == 1]
df_inc_all  = df_responses[df_responses.feedback == -1]

# --- Contrast order ---
contrasts_sorted = sorted(df_responses['contrast'].unique())

# --- Colors ---
color_left  = "#F9A65A"   # pastel orange
color_right = "#69BDBF"   # pastel teal

# --- Create 2×3 figure ---
fig, axes = plt.subplots(
    2, 3,
    figsize=(22, 10),
    sharey=True,
    sharex=True,
    gridspec_kw={'wspace': 0.25, 'hspace': 0.20}
)

datasets = [
    (df_corr_all, 0, "Correct trials"),
    (df_inc_all,  1, "Incorrect trials")
]

# ============================================================
# MAIN LOOP
# ============================================================
for df_main, row_idx, row_title in datasets:
    for col_idx, p_left in enumerate(p_left_values):

        ax = axes[row_idx, col_idx]

        # Filter by p_left
        df_p = df_main[df_main.p_left == p_left]

        # Build RT arrays for each contrast AND each choice
        rts_left = []
        rts_right = []

        for c in contrasts_sorted:
            df_c = df_p[df_p.contrast == c]

            rt_left  = np.log10(df_c[df_c.choice == -1]['reaction_time'].dropna().values)
            rt_right = np.log10(df_c[df_c.choice == 1 ]['reaction_time'].dropna().values)

            rts_left.append(rt_left)
            rts_right.append(rt_right)

        # ----------------------------  
        # Positions for side-by-side violins
        # ----------------------------
        pos = np.arange(1, len(contrasts_sorted) + 1)
        pos_left  = pos - 0.18
        pos_right = pos + 0.18

        # ----------------------------  
        # Left choice violins  
        # ----------------------------
        vp_left = ax.violinplot(
            rts_left,
            positions=pos_left,
            showextrema=False,
            showmedians=True,
            vert=False
        )
        for b in vp_left['bodies']:
            b.set_facecolor(color_left)
            b.set_alpha(0.65)
        vp_left['cmedians'].set_color(color_left)

        # ----------------------------  
        # Right choice violins  
        # ----------------------------
        vp_right = ax.violinplot(
            rts_right,
            positions=pos_right,
            showextrema=False,
            showmedians=True,
            vert=False
        )
        for b in vp_right['bodies']:
            b.set_facecolor(color_right)
            b.set_alpha(0.65)
        vp_right['cmedians'].set_color(color_right)

        # ----------------------------  
        # Median text  
        # ----------------------------
        for i, (rtL, rtR) in enumerate(zip(rts_left, rts_right)):
            if len(rtL) > 0:
                mL = np.median(rtL)
                ax.text(mL, pos_left[i], f'{10**mL:.2f}s',
                        ha='left', va='bottom', rotation=45, size=13)
            if len(rtR) > 0:
                mR = np.median(rtR)
                ax.text(mR, pos_right[i], f'{10**mR:.2f}s',
                        ha='left', va='bottom', rotation=45, size=13)

        # ----------------------------  
        # Axis formatting  
        # ----------------------------
        ax.set_yticks(pos)
        ax.set_yticklabels([f'{c*100:.0f}' for c in contrasts_sorted])

        if col_idx == 0:
            ax.set_ylabel("Contrast")
        else:
            ax.set_ylabel("")

        xticks = np.linspace(-2, 2, 6)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'$10^{int(t)}$' for t in xticks])

        # Remove x-axis labels/ticks for FIRST ROW (Correct trials)
        if row_idx == 0:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Reaction time (s)")


        ax.set_title(f"{row_title} | p_left = {p_left}", fontsize=16, pad=20)

        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        clip_axes_to_ticks(ax=ax)


patch_left  = mpatches.Patch(color=color_left,  label='Left choice')
patch_right = mpatches.Patch(color=color_right, label='Right choice')
fig.legend(
    handles=[patch_left, patch_right],
    loc='center right',
    bbox_to_anchor=(1.08, 0.5),
    frameon=False,
    fontsize=16
)

plt.tight_layout()
set_plotsize(w=30, h=21)


#%%
"""
easier way to analyse
Left-Right
positive values are a bias in a shorter RT to the right
negative values are a bias in a shorter RT to the left  
1x3 for each biased block
correct blue
incorrect red
""" 
p_left_values = sorted(df_responses.p_left.unique())   # [0.2, 0.5, 0.8]
contrasts_sorted = sorted(df_responses['contrast'].unique())

color_correct   = "#1A94FF"   # teal
color_incorrect = "#FF3546"   # orange

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 5),
    sharey=True,
    gridspec_kw={'wspace': 0.25}
)

for col_idx, p_left in enumerate(p_left_values):

    ax = axes[col_idx]

    # Filter block
    df_block = df_responses[df_responses.p_left == p_left]

    diff_correct = []
    diff_incorrect = []

    for c in contrasts_sorted:

        df_c = df_block[df_block.contrast == c]

        df_c_cor = df_c[df_c.feedback == 1]
        df_c_inc = df_c[df_c.feedback == -1]

        # Compute medians safely
        def get_med(df, choice):
            vals = np.log10(df[df.choice == choice]['reaction_time'].dropna().values)
            return np.median(vals) if len(vals) > 0 else np.nan

        m_cor_L = get_med(df_c_cor, -1)
        m_cor_R = get_med(df_c_cor,  1)

        m_inc_L = get_med(df_c_inc, -1)
        m_inc_R = get_med(df_c_inc,  1)

        # Left - Right difference
        diff_correct.append(m_cor_L- m_cor_R)
        diff_incorrect.append(m_inc_L - m_inc_R)

    # X-axis values
    x = np.arange(len(contrasts_sorted))

    # PLOT CORRECT
    ax.plot(
        x, diff_correct,
        marker='o', color=color_correct,
        label='Correct'
    )

    # PLOT INCORRECT
    ax.plot(
        x, diff_incorrect,
        marker='o', color=color_incorrect,
        label='Incorrect'
    )

    # Formatting
    ax.axhline(0, color='black', linewidth=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c*100:.0f}' for c in contrasts_sorted])
    ax.set_xlabel("Contrast (%)")

    if col_idx == 0:
        ax.set_ylabel("Δ median RT (log10 s)\nLeft – Right")

    ax.set_title(f"Block p_left = {p_left}", fontsize=14, pad=18)

fig.legend(
    loc='center right',
    bbox_to_anchor=(1.06, 0.5),
    frameon=False,
    fontsize=14
)

plt.tight_layout()
set_plotsize(w=22, h=6)


#%% 
"""
#########################################################################
plot the same but split by mouse 
""" 
subjects = sorted(df_responses.subject.unique())

for subj in subjects:

    # Filter subject
    df_subj = df_responses[df_responses.subject == subj]

    p_left_values = sorted(df_subj.p_left.unique())
    contrasts_sorted = sorted(df_subj['contrast'].unique())

    color_correct   = "#1A94FF"   # blue
    color_incorrect = "#FF3546"   # red

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5),
        sharey=True,
        gridspec_kw={'wspace': 0.25}
    )

    for col_idx, p_left in enumerate(p_left_values):

        ax = axes[col_idx]

        # Filter block
        df_block = df_subj[df_subj.p_left == p_left]

        diff_correct = []
        diff_incorrect = []

        for c in contrasts_sorted:

            df_c = df_block[df_block.contrast == c]

            df_c_cor = df_c[df_c.feedback == 1]
            df_c_inc = df_c[df_c.feedback == -1]

            # safe median function (your original)
            def get_med(df, choice):
                vals = np.log10(df[df.choice == choice]['reaction_time'].dropna().values)
                return np.median(vals) if len(vals) > 0 else np.nan

            m_cor_L = get_med(df_c_cor, -1)
            m_cor_R = get_med(df_c_cor,  1)

            m_inc_L = get_med(df_c_inc, -1)
            m_inc_R = get_med(df_c_inc,  1)

            # Left - Right difference
            diff_correct.append(m_cor_L - m_cor_R)
            diff_incorrect.append(m_inc_L - m_inc_R)

        # X positions
        x = np.arange(len(contrasts_sorted))

        # PLOT CORRECT
        ax.plot(
            x, diff_correct,
            marker='o', color=color_correct,
            label='Correct'
        )

        # PLOT INCORRECT
        ax.plot(
            x, diff_incorrect,
            marker='o', color=color_incorrect,
            label='Incorrect'
        )

        # Formatting
        ax.axhline(0, color='black', linewidth=1, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c*100:.0f}' for c in contrasts_sorted])
        ax.set_xlabel("Contrast (%)")

        if col_idx == 0:
            ax.set_ylabel("Δ median RT (log10 s)\nLeft – Right")

        ax.set_title(f"{subj} | Block p_left = {p_left}", fontsize=14, pad=18)

    fig.legend(
        loc='center right',
        bbox_to_anchor=(1.06, 0.5),
        frameon=False,
        fontsize=14
    )

    plt.tight_layout()
    set_plotsize(w=22, h=6)
    plt.show()



#%% 
"""
#########################################################################
same but only 0.5 
""" 
subjects = sorted(df_responses.subject.unique())

for subj in subjects:

    # Filter subject
    df_subj = df_responses[df_responses.subject == subj]

    # Only unbiased 0.5 block
    p_left_values = [0.5]

    contrasts_sorted = sorted(df_subj['contrast'].unique())

    color_correct   = "#1A94FF"   # blue
    color_incorrect = "#FF3546"   # red

    fig, ax = plt.subplots(
        1, 1,
        figsize=(6, 5),
        sharey=True
    )

    p_left = 0.5
    df_block = df_subj[df_subj.p_left == p_left]

    diff_correct = []
    diff_incorrect = []

    for c in contrasts_sorted:

        df_c = df_block[df_block.contrast == c]

        df_c_cor = df_c[df_c.feedback == 1]
        df_c_inc = df_c[df_c.feedback == -1]

        def get_med(df, choice):
            vals = np.log10(df[df.choice == choice]['reaction_time'].dropna().values)
            return np.median(vals) if len(vals) > 0 else np.nan

        m_cor_L = get_med(df_c_cor, -1)
        m_cor_R = get_med(df_c_cor,  1)

        m_inc_L = get_med(df_c_inc, -1)
        m_inc_R = get_med(df_c_inc,  1)

        # Left - Right difference
        diff_correct.append(m_cor_L - m_cor_R)
        diff_incorrect.append(m_inc_L - m_inc_R)

    # X positions
    x = np.arange(len(contrasts_sorted))

    # PLOT CORRECT
    ax.plot(
        x, diff_correct,
        marker='o', color=color_correct,
        label='Correct'
    )

    # PLOT INCORRECT
    ax.plot(
        x, diff_incorrect,
        marker='o', color=color_incorrect,
        label='Incorrect'
    )

    # Formatting
    ax.axhline(0, color='black', linewidth=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c*100:.0f}' for c in contrasts_sorted])
    ax.set_xlabel("Contrast (%)")
    # ax.set_ylabel("Δ median RT (log10 s)\nLeft – Right")
    ax.set_title(f"{subj} | Block p_left = 0.5", fontsize=14, pad=18)

    plt.tight_layout()
    set_plotsize(w=8, h=6)
    plt.show() 



#%%
"""
Δ RT = median RT(correct) - median RT(incorrect)
Positive values → incorrect faster (more impulsive)
Negative values → correct faster (more cautious)

One row: subjects (columns)
Two lines per subject: choice -1 (Left) and choice +1 (Right)
"""

subjects = sorted(df_responses.subject.unique())
n_subj = len(subjects)

fig, axes = plt.subplots(
    1, n_subj,
    figsize=(5 * n_subj, 5),
    sharey=True,
    gridspec_kw={'wspace': 0.25}
)

if n_subj == 1:
    axes = [axes]

color_left  = "#F67B00"   # choice -1
color_right = "#0A9396"   # choice +1

for idx, subj in enumerate(subjects):

    ax = axes[idx]

    # Filter subject
    df_subj = df_responses[df_responses.subject == subj]

    # Only unbiased block p_left = 0.5
    p_left = 0.5
    df_block = df_subj[df_subj.p_left == p_left]

    contrasts_sorted = sorted(df_block['contrast'].unique())

    # Will hold ΔRT(correct - incorrect) for each choice and contrast
    diff_left  = []   # choice = -1
    diff_right = []   # choice =  1

    for c in contrasts_sorted:

        df_c = df_block[df_block.contrast == c]

        def get_median_rt(df, fb, choice):
            vals = np.log10(
                df[(df.feedback == fb) & (df.choice == choice)]['reaction_time'].dropna().values
            )
            return np.median(vals) if len(vals) > 0 else np.nan

        # Left choice (-1)
        m_cor_L = get_median_rt(df_c, fb=1,  choice=-1)
        m_inc_L = get_median_rt(df_c, fb=-1, choice=-1)
        diff_left.append(m_cor_L - m_inc_L)

        # Right choice (+1)
        m_cor_R = get_median_rt(df_c, fb=1,  choice=1)
        m_inc_R = get_median_rt(df_c, fb=-1, choice=1)
        diff_right.append(m_cor_R - m_inc_R)

    # X positions
    x = np.arange(len(contrasts_sorted))

    # Plot Left choice (-1)
    ax.plot(
        x, diff_left,
        marker='o', color=color_left,
        label='Choice Left (-1)'
    )

    # Plot Right choice (+1)
    ax.plot(
        x, diff_right,
        marker='o', color=color_right,
        label='Choice Right (+1)'
    )

    # Formatting
    ax.axhline(0, color='black', linewidth=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c*100:.0f}' for c in contrasts_sorted])
    ax.set_xlabel("Contrast (%)")

    if idx == 0:
        ax.set_ylabel("Δ median RT (log10 s)\nCorrect − Incorrect")

    ax.set_title(f"{subj}\nBlock p_left = 0.5", fontsize=14, pad=18)

# # Optional shared legend
# fig.legend(
#     loc='center right',
#     bbox_to_anchor=(1.03, 0.5),
#     frameon=False,
#     fontsize=12
# )

plt.tight_layout()
set_plotsize(w=5 * n_subj + 2, h=4)
plt.show()





#%% 
"""
#########################################################################
5-HT ACTIVITY PLOTTEEEEEEED 
"""
"""
NM ΔResponse Left - Right
Per mouse, per bias block (p_left), per contrast
Split by correct / incorrect feedback
"""

subjects = sorted(df_responses.subject.unique())

color_correct   = "#1A94FF"   # blue  (correct)
color_incorrect = "#FF3546"   # red   (incorrect)

for subj in subjects:

    # Filter subject
    df_subj = df_responses[df_responses.subject == subj]

    p_left_values = sorted(df_subj.p_left.unique())
    contrasts_sorted = sorted(df_subj['contrast'].unique())

    fig, axes = plt.subplots(
        1, len(p_left_values),
        figsize=(18, 5),
        sharey=True,
        gridspec_kw={'wspace': 0.25}
    )

    if len(p_left_values) == 1:
        axes = [axes]

    for col_idx, p_left in enumerate(p_left_values):

        ax = axes[col_idx]

        # Filter this block
        df_block = df_subj[df_subj.p_left == p_left]

        diff_correct = []
        diff_incorrect = []

        for c in contrasts_sorted:

            df_c = df_block[df_block.contrast == c]

            # correct and incorrect
            df_cor = df_c[df_c.feedback == 1]
            df_inc = df_c[df_c.feedback == -1]

            # small helper function
            def get_med_nm(df, choice):
                vals = df[df.choice == choice]["response_mean"].dropna().values
                return np.median(vals) if len(vals) > 0 else np.nan

            # LEFT = -1, RIGHT = 1

            # Correct trials
            m_cor_L = get_med_nm(df_cor, -1)
            m_cor_R = get_med_nm(df_cor, 1)

            # Incorrect trials
            m_inc_L = get_med_nm(df_inc, -1)
            m_inc_R = get_med_nm(df_inc, 1)

            diff_correct.append(m_cor_L - m_cor_R)
            diff_incorrect.append(m_inc_L - m_inc_R)

        # X axis
        x = np.arange(len(contrasts_sorted))

        # PLOT Correct NM L–R
        ax.plot(
            x, diff_correct,
            marker='o', color=color_correct,
            label='Correct'
        )

        # PLOT Incorrect NM L–R
        ax.plot(
            x, diff_incorrect,
            marker='o', color=color_incorrect,
            label='Incorrect'
        )

        # Formatting
        ax.axhline(0, color='black', alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c*100:.0f}" for c in contrasts_sorted])
        ax.set_xlabel("Contrast (%)")

        if col_idx == 0:
            ax.set_ylabel("Δ NM response\nLeft – Right")

        ax.set_title(f"{subj} | p_left = {p_left}", fontsize=14)

    fig.legend(
        loc='center right',
        bbox_to_anchor=(1.06, 0.5),
        frameon=False,
        fontsize=14
    )

    plt.tight_layout()
    set_plotsize(w=22, h=6)
    plt.show()







#%%
""" 
#########################################################################
all subplots in one for the diff mice and p=0.5
"""
subjects = sorted(df_responses.subject.unique())
n_subj = len(subjects)

# Create 1 row × n_subj columns
fig, axes = plt.subplots(
    1, n_subj,
    figsize=(5 * n_subj, 5),   # scale width dynamically
    sharey=True,
    gridspec_kw={'wspace': 0.25}
)

# If only 1 subject, axes is not a list → make it one
if n_subj == 1:
    axes = [axes]

color_correct   = "#1A94FF"   # blue
color_incorrect = "#FF3546"   # red

for idx, subj in enumerate(subjects):

    ax = axes[idx]

    # Filter subject
    df_subj = df_responses[df_responses.subject == subj]

    # Only unbiased block 0.5
    p_left = 0.5
    contrasts_sorted = sorted(df_subj['contrast'].unique())

    df_block = df_subj[df_subj.p_left == p_left]

    diff_correct = []
    diff_incorrect = []

    for c in contrasts_sorted:

        df_c = df_block[df_block.contrast == c]

        df_c_cor = df_c[df_c.feedback == 1]
        df_c_inc = df_c[df_c.feedback == -1]

        # safe median function
        def get_med(df, choice):
            vals = np.log10(df[df.choice == choice]['reaction_time'].dropna().values)
            return np.median(vals) if len(vals) > 0 else np.nan

        m_cor_L = get_med(df_c_cor, -1)
        m_cor_R = get_med(df_c_cor,  1)

        m_inc_L = get_med(df_c_inc, -1)
        m_inc_R = get_med(df_c_inc,  1)

        diff_correct.append(m_cor_L - m_cor_R)
        diff_incorrect.append(m_inc_L - m_inc_R)

    # X positions
    x = np.arange(len(contrasts_sorted))

    # PLOT CORRECT
    ax.plot(
        x, diff_correct,
        marker='o', color=color_correct,
        label='Correct'
    )

    # PLOT INCORRECT
    ax.plot(
        x, diff_incorrect,
        marker='o', color=color_incorrect,
        label='Incorrect'
    )

    # Formatting
    ax.axhline(0, color='black', linewidth=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c*100:.0f}' for c in contrasts_sorted])
    ax.set_xlabel("Contrast (%)")

    if idx == 0:
        ax.set_ylabel("Δ median RT (log10 s)\nLeft – Right")

    ax.set_title(f"{subj}\nBlock p_left = 0.5", fontsize=14, pad=18)

# # Shared legend for all subjects
# fig.legend(
#     loc='center right',
#     bbox_to_anchor=(1.03, 0.5),
#     frameon=False,
#     fontsize=14
# )

plt.tight_layout()
set_plotsize(w=5 * n_subj + 2, h=4)
plt.show()




#%%
"""
#########################################################################
KDE RT plots per mouse for unbiased blocks for each contrast, left right
"""
import seaborn as sns 

subjects = sorted(df_responses.subject.unique())
contrasts_sorted = sorted(df_responses['contrast'].unique())

color_left  = "#F67B00"  
color_right = "#0A9396"  

for subj in subjects:

    # *** ONLY unbiased block ***
    df_subj = df_responses[(df_responses.subject == subj) &
                           (df_responses.p_left == 0.5)]

    n_contrasts = len(contrasts_sorted)

    fig, axes = plt.subplots(
        2, n_contrasts,
        figsize=(4 * n_contrasts, 8),
        sharey='row',
        sharex=True,
        gridspec_kw={'hspace': 0.35, 'wspace': 0.25}
    )

    row_titles = ["Correct", "Incorrect"]
    fb_values  = [1, -1]

    for row_idx, fb in enumerate(fb_values):

        df_fb = df_subj[df_subj.feedback == fb]

        for col_idx, c in enumerate(contrasts_sorted):

            ax = axes[row_idx, col_idx]
            df_c = df_fb[df_fb.contrast == c]

            # Split LEFT/RIGHT by 'choice' (True=Left, False=Right)
            rt_left  = np.log10(df_c[df_c.choice == -1 ]['reaction_time'].dropna())
            rt_right = np.log10(df_c[df_c.choice == 1]['reaction_time'].dropna())

            if len(rt_left) > 1:
                sns.kdeplot(rt_left, ax=ax,
                            color=color_left, fill=True,
                            alpha=0.4, linewidth=2, label="Left")

            if len(rt_right) > 1:
                sns.kdeplot(rt_right, ax=ax,
                            color=color_right, fill=True,
                            alpha=0.4, linewidth=2, label="Right")

            if row_idx == 0:
                ax.set_title(f"Contrast {c*100:.0f}%", fontsize=12)

            if col_idx == 0:
                ax.set_ylabel(f"{row_titles[row_idx]}\nDensity")

            ax.set_xlabel("log10 RT (s)")

    # fig.legend(
    #     loc='upper right',
    #     bbox_to_anchor=(1.02, 1.0),
    #     frameon=False,
    #     fontsize=12
    # )

    fig.suptitle(f"RT density by contrast and choice - {subj} (p_left = 0.5)", 
                 fontsize=15, y=1.03)

    plt.tight_layout()
    set_plotsize(w=4 * n_contrasts + 1, h=9)
    plt.show()







#%%
"""
########################################################################
By just changing the target
plot all of the plots above
be careful with which event to pick
    one of them for the RTs
    by each one if analysing the NM responses 
"""
TARGET = 'NBM'
df_responses = df_original[df_original.event == 'feedback_times'].copy()
df_5HT = df_responses[df_responses.target == TARGET]
df_responses = df_5HT.copy()

# Plot log reaction time distributions for each contrast level
rts = [
    np.log10(t['reaction_time'].dropna().values)
    for _, t in df_responses.groupby('contrast')
    ]
fig, ax = plt.subplots()
parts = ax.violinplot(
    rts,
    showextrema=False,
    showmedians=True,
    # orientation='horizontal' #giving error "TypeError: Axes.violinplot() got an unexpected keyword argument 'orientation' 10122025" 
    vert=False
    )
cmap = plt.colormaps['Greys']
rt_colors = cmap(np.linspace(0.4, 0.99, len(rts)))
for pc, color in zip(parts['bodies'], rt_colors):
    pc.set_facecolor(color)
parts['cmedians'].set_colors(rt_colors)
for i, rt in enumerate(rts):
    median = np.median(rt)
    ax.text(
        median, i + 1, f'{10**median:.2f}s', rotation=45, ha='left', va='bottom', size=14
        )
contrasts = df_responses['contrast'].unique()
ax.set_yticks(np.arange(1, len(contrasts) + 1))
ax.set_yticklabels([f'{c*100:.0f}' for c in sorted(contrasts)])
ax.set_ylabel('Contrast level')
xticks = np.linspace(-2, 2, 6)
ax.set_xticks(xticks)
ax.set_xticklabels(['$10^{%d}$' % t for t in xticks])
ax.set_xlim([-1.5, 1.5])
ax.set_title(
    f"Reaction Time Distributions by Contrast Level {TARGET}",
    fontsize=16,
    fontweight="bold",
    pad=31     # space between title and plot
)
ax.set_xlabel('Reaction time (s)')
clip_axes_to_ticks(ax=ax)
set_plotsize(w=10, h=5, ax=ax)
# fig.savefig('figures/reaction_times.svg')
# --- Split dataframe ---
df_corr = df_responses[df_responses.feedback == 1]
df_inc  = df_responses[df_responses.feedback == -1]

# --- Prepare contrasts ---
contrasts_sorted = sorted(df_responses['contrast'].unique())

# --- Create figure with 2 subplots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

datasets = [
    (df_corr, axes[0], f"Correct trials {TARGET}"),
    (df_inc,  axes[1], f"Incorrect trials {TARGET}")]

for df, ax, title in datasets:
    # --- RT distributions per contrast ---
    rts = [
        np.log10(t['reaction_time'].dropna().values)
        for _, t in df.groupby('contrast') ]

    # --- Violin plot ---
    parts = ax.violinplot(
        rts,
        showextrema=False,
        showmedians=True,
        # orientation='horizontal' #giving error "TypeError: Axes.violinplot() got an unexpected keyword argument 'orientation' 10122025" 
        vert=False)

    # --- Color map ---
    cmap = plt.colormaps['Greys']
    rt_colors = cmap(np.linspace(0.4, 0.99, len(rts)))

    for pc, color in zip(parts['bodies'], rt_colors):
        pc.set_facecolor(color)
    parts['cmedians'].set_colors(rt_colors)

    # --- Add median text ---
    for i, rt in enumerate(rts):
        if len(rt) == 0:
            continue
        median = np.median(rt)
        ax.text(
            median, i + 1, 
            f'{10**median:.2f}s',
            rotation=45,
            ha='left', va='bottom', size=14)

    # --- Y-axis labels ---
    ax.set_yticks(np.arange(1, len(contrasts_sorted) + 1))
    ax.set_yticklabels([f'{c*100:.0f}' for c in contrasts_sorted])
    ax.set_ylabel('Contrast level')

    # --- X-axis ticks ---
    xticks = np.linspace(-2, 2, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'$10^{int(t)}$' for t in xticks])
    # ax.set_xlim([-2.1, 2])
    ax.set_xlabel('Reaction time (s)')

    # --- Title ---
    ax.set_title(title)
    ax.set_title(
        title, 
        fontsize=16,
        # fontweight="bold",
        pad=31     # space between title and plot
    )
    clip_axes_to_ticks(ax=ax)

plt.tight_layout()
set_plotsize(w=25, h=8) 















df_corr = df_responses[df_responses.feedback == 1]
df_inc  = df_responses[df_responses.feedback == -1]

# Unique and sorted p_left values
p_left_values = sorted(df_responses.p_left.unique())   # [0.2, 0.5, 0.8]

# Prepare contrasts
contrasts_sorted = sorted(df_responses['contrast'].unique())

# Create figure with 2 rows × 3 columns
fig, axes = plt.subplots(
    2, 3,
    figsize=(22, 8),
    sharey=True,
    sharex=True, 
    gridspec_kw={'hspace': 0.55}   # increase vertical spacing 
)

datasets = [
    (df_corr, 0, "Correct trials"),    # first row
    (df_inc,  1, "Incorrect trials"),  # second row
]


for df_main, row_idx, row_title in datasets:

    for col_idx, p_left in enumerate(p_left_values):

        ax = axes[row_idx, col_idx]

        # Filter by p_left
        df = df_main[df_main.p_left == p_left]

        # RT distributions per contrast
        rts = [
            np.log10(t['reaction_time'].dropna().values)
            for _, t in df.groupby('contrast')
        ]

        # Violin plot
        parts = ax.violinplot(
            rts,
            showextrema=False,
            showmedians=True,
            vert=False
        )

        # Colors
        cmap = plt.colormaps['Greys']
        rt_colors = cmap(np.linspace(0.4, 0.99, len(rts)))

        for pc, color in zip(parts['bodies'], rt_colors):
            pc.set_facecolor(color)
        parts['cmedians'].set_colors(rt_colors)

        # Add medians
        for i, rt in enumerate(rts):
            if len(rt) == 0:
                continue
            median = np.median(rt)
            ax.text(
                median, i + 1,
                f'{10**median:.2f}s',
                rotation=45,
                ha='left', va='bottom',
                size=12
            )

        # -------------------------------
        # Y-axis only on the first column
        # -------------------------------
        ax.set_yticks(np.arange(1, len(contrasts_sorted) + 1))
        ax.set_yticklabels([f'{c*100:.0f}' for c in contrasts_sorted])

        if col_idx == 0:
            # First column → keep the axis label
            ax.set_ylabel('Contrast')
        else:
            # Other columns → keep tick values, remove axis label
            ax.set_ylabel('')


        # -------------------------------
        # X-axis only on the second row
        # -------------------------------
        xticks = np.linspace(-2, 2, 6)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'$10^{int(t)}$' for t in xticks])

        if row_idx == 0:
            # Remove x-axis labels/ticks for the first row
            ax.set_xlabel('')
            ax.set_xticklabels([])

        else:
            # Second row → show x-axis label
            ax.set_xlabel('Reaction time (s)')

        # Title for each subplot
        ax.set_title(
            f'{row_title} | p_left = {p_left}',
            fontsize=14,
            pad=25
        )

        clip_axes_to_ticks(ax=ax)


plt.tight_layout()
set_plotsize(w=28, h=10)


"""
#########################################################################
only DR
only unbiased block
RT split by correct/incorrect and choice side left/right 
""" 
import matplotlib.patches as mpatches

df_unbiased = df_responses[df_responses.p_left == 0.5]

df_corr = df_unbiased[df_unbiased.feedback == 1]
df_inc  = df_unbiased[df_unbiased.feedback == -1]

contrasts_sorted = sorted(df_unbiased['contrast'].unique())

color_left  = "#F9A65A"   # pastel orange (based on #F67B00)
color_right = "#69BDBF"   # pastel teal   (based on #0A9396)

fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 5),
    sharey=True,
    gridspec_kw={'wspace': 0.15})

datasets = [
    (df_corr, axes[0], "Correct trials"),
    (df_inc,  axes[1], "Incorrect trials")]

for df_main, ax, title in datasets:
    # Build RT arrays for each contrast AND each choice
    rts_left = []
    rts_right = []

    for c in contrasts_sorted:
        df_c = df_main[df_main.contrast == c]

        rt_left  = np.log10(df_c[df_c.choice == -1]['reaction_time'].dropna().values)
        rt_right = np.log10(df_c[df_c.choice == 1 ]['reaction_time'].dropna().values)

        rts_left.append(rt_left)
        rts_right.append(rt_right)

    # ---------------------------------------------------------
    # Plot violins side by side for left/right choices
    # ---------------------------------------------------------

    # Shift positions for left/right violins:
    pos = np.arange(1, len(contrasts_sorted) + 1)
    pos_left  = pos - 0.18
    pos_right = pos + 0.18

    # Left choice violins
    vp_left = ax.violinplot(
        rts_left,
        positions=pos_left,
        showextrema=False,
        showmedians=True,
        vert=False
    )
    for b in vp_left['bodies']:
        b.set_facecolor(color_left)
        b.set_alpha(0.65)
    vp_left['cmedians'].set_color(color_left)

    # Right choice violins
    vp_right = ax.violinplot(
        rts_right,
        positions=pos_right,
        showextrema=False,
        showmedians=True,
        vert=False
    )
    for b in vp_right['bodies']:
        b.set_facecolor(color_right)
        b.set_alpha(0.65)
    vp_right['cmedians'].set_color(color_right)

    # ---------------------------------------------------------
    # Add median text for both choices
    # ---------------------------------------------------------
    for i, (rtL, rtR) in enumerate(zip(rts_left, rts_right)):
        if len(rtL) > 0:
            mL = np.median(rtL)
            ax.text(mL, pos_left[i], f'{10**mL:.2f}s',
                    ha='left', va='bottom', rotation=45, size=15)
        if len(rtR) > 0:
            mR = np.median(rtR)
            ax.text(mR, pos_right[i], f'{10**mR:.2f}s',
                    ha='left', va='bottom', rotation=45, size=15)

    # ---------------------------------------------------------
    # Axes 
    # ---------------------------------------------------------
    ax.set_yticks(pos)
    ax.set_yticklabels([f'{c*100:.0f}' for c in contrasts_sorted])

    if col_idx == 0:
        # First column → keep the axis label
        ax.set_ylabel('Contrast')
    else:
        # Other columns → keep tick values, remove axis label
        ax.set_ylabel('')

    xticks = np.linspace(-2, 2, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'$10^{int(t)}$' for t in xticks])
    ax.set_xlabel("Reaction time (s)")

    ax.set_title(title, fontsize=16, pad=20)

    patch_left  = mpatches.Patch(color=color_left,  label='Left choice')
    patch_right = mpatches.Patch(color=color_right, label='Right choice')
    fig.legend(
        handles=[patch_left, patch_right],
        loc='center right',
        bbox_to_anchor=(1.04, 0.5),   # move legend outside the figure
        frameon=False,
        fontsize=14)
        
    clip_axes_to_ticks(ax=ax)

plt.tight_layout()
set_plotsize(w=20, h=10)



"""
##########################################################################
same but split by the probabilityLeft blocks 
2x3 grid
""" 
# --- Sort p_left values ---
p_left_values = sorted(df_responses.p_left.unique())   # [0.2, 0.5, 0.8]

df_corr_all = df_responses[df_responses.feedback == 1]
df_inc_all  = df_responses[df_responses.feedback == -1]

# --- Contrast order ---
contrasts_sorted = sorted(df_responses['contrast'].unique())

# --- Colors ---
color_left  = "#F9A65A"   # pastel orange
color_right = "#69BDBF"   # pastel teal

# --- Create 2×3 figure ---
fig, axes = plt.subplots(
    2, 3,
    figsize=(22, 10),
    sharey=True,
    sharex=True,
    gridspec_kw={'wspace': 0.25, 'hspace': 0.20}
)

datasets = [
    (df_corr_all, 0, "Correct trials"),
    (df_inc_all,  1, "Incorrect trials")
]

# ============================================================
# MAIN LOOP
# ============================================================
for df_main, row_idx, row_title in datasets:
    for col_idx, p_left in enumerate(p_left_values):

        ax = axes[row_idx, col_idx]

        # Filter by p_left
        df_p = df_main[df_main.p_left == p_left]

        # Build RT arrays for each contrast AND each choice
        rts_left = []
        rts_right = []

        for c in contrasts_sorted:
            df_c = df_p[df_p.contrast == c]

            rt_left  = np.log10(df_c[df_c.choice == -1]['reaction_time'].dropna().values)
            rt_right = np.log10(df_c[df_c.choice == 1 ]['reaction_time'].dropna().values)

            rts_left.append(rt_left)
            rts_right.append(rt_right)

        # ----------------------------  
        # Positions for side-by-side violins
        # ----------------------------
        pos = np.arange(1, len(contrasts_sorted) + 1)
        pos_left  = pos - 0.18
        pos_right = pos + 0.18

        # ----------------------------  
        # Left choice violins  
        # ----------------------------
        vp_left = ax.violinplot(
            rts_left,
            positions=pos_left,
            showextrema=False,
            showmedians=True,
            vert=False
        )
        for b in vp_left['bodies']:
            b.set_facecolor(color_left)
            b.set_alpha(0.65)
        vp_left['cmedians'].set_color(color_left)

        # ----------------------------  
        # Right choice violins  
        # ----------------------------
        vp_right = ax.violinplot(
            rts_right,
            positions=pos_right,
            showextrema=False,
            showmedians=True,
            vert=False
        )
        for b in vp_right['bodies']:
            b.set_facecolor(color_right)
            b.set_alpha(0.65)
        vp_right['cmedians'].set_color(color_right)

        # ----------------------------  
        # Median text  
        # ----------------------------
        for i, (rtL, rtR) in enumerate(zip(rts_left, rts_right)):
            if len(rtL) > 0:
                mL = np.median(rtL)
                ax.text(mL, pos_left[i], f'{10**mL:.2f}s',
                        ha='left', va='bottom', rotation=45, size=13)
            if len(rtR) > 0:
                mR = np.median(rtR)
                ax.text(mR, pos_right[i], f'{10**mR:.2f}s',
                        ha='left', va='bottom', rotation=45, size=13)

        # ----------------------------  
        # Axis formatting  
        # ----------------------------
        ax.set_yticks(pos)
        ax.set_yticklabels([f'{c*100:.0f}' for c in contrasts_sorted])

        if col_idx == 0:
            ax.set_ylabel("Contrast")
        else:
            ax.set_ylabel("")

        xticks = np.linspace(-2, 2, 6)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'$10^{int(t)}$' for t in xticks])

        # Remove x-axis labels/ticks for FIRST ROW (Correct trials)
        if row_idx == 0:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Reaction time (s)")


        ax.set_title(f"{row_title} | p_left = {p_left}", fontsize=16, pad=20)

        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        clip_axes_to_ticks(ax=ax)


patch_left  = mpatches.Patch(color=color_left,  label='Left choice')
patch_right = mpatches.Patch(color=color_right, label='Right choice')
fig.legend(
    handles=[patch_left, patch_right],
    loc='center right',
    bbox_to_anchor=(1.08, 0.5),
    frameon=False,
    fontsize=16
)

plt.tight_layout()
set_plotsize(w=30, h=21)


"""
easier way to analyse
Left-Right
positive values are a bias in a shorter RT to the right
negative values are a bias in a shorter RT to the left  
1x3 for each biased block
correct blue
incorrect red
""" 
p_left_values = sorted(df_responses.p_left.unique())   # [0.2, 0.5, 0.8]
contrasts_sorted = sorted(df_responses['contrast'].unique())

color_correct   = "#1A94FF"   # teal
color_incorrect = "#FF3546"   # orange

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 5),
    sharey=True,
    gridspec_kw={'wspace': 0.25}
)

for col_idx, p_left in enumerate(p_left_values):

    ax = axes[col_idx]

    # Filter block
    df_block = df_responses[df_responses.p_left == p_left]

    diff_correct = []
    diff_incorrect = []

    for c in contrasts_sorted:

        df_c = df_block[df_block.contrast == c]

        df_c_cor = df_c[df_c.feedback == 1]
        df_c_inc = df_c[df_c.feedback == -1]

        # Compute medians safely
        def get_med(df, choice):
            vals = np.log10(df[df.choice == choice]['reaction_time'].dropna().values)
            return np.median(vals) if len(vals) > 0 else np.nan

        m_cor_L = get_med(df_c_cor, -1)
        m_cor_R = get_med(df_c_cor,  1)

        m_inc_L = get_med(df_c_inc, -1)
        m_inc_R = get_med(df_c_inc,  1)

        # Left - Right difference
        diff_correct.append(m_cor_L- m_cor_R)
        diff_incorrect.append(m_inc_L - m_inc_R)

    # X-axis values
    x = np.arange(len(contrasts_sorted))

    # PLOT CORRECT
    ax.plot(
        x, diff_correct,
        marker='o', color=color_correct,
        label='Correct'
    )

    # PLOT INCORRECT
    ax.plot(
        x, diff_incorrect,
        marker='o', color=color_incorrect,
        label='Incorrect'
    )

    # Formatting
    ax.axhline(0, color='black', linewidth=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c*100:.0f}' for c in contrasts_sorted])
    ax.set_xlabel("Contrast (%)")

    if col_idx == 0:
        ax.set_ylabel("Δ median RT (log10 s)\nLeft – Right")

    ax.set_title(f"Block p_left = {p_left}", fontsize=14, pad=18)

fig.legend(
    loc='center right',
    bbox_to_anchor=(1.06, 0.5),
    frameon=False,
    fontsize=14
)

plt.tight_layout()
set_plotsize(w=22, h=6)


"""
#########################################################################
plot the same but split by mouse 
""" 
subjects = sorted(df_responses.subject.unique())

for subj in subjects:

    # Filter subject
    df_subj = df_responses[df_responses.subject == subj]

    p_left_values = sorted(df_subj.p_left.unique())
    contrasts_sorted = sorted(df_subj['contrast'].unique())

    color_correct   = "#1A94FF"   # blue
    color_incorrect = "#FF3546"   # red

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5),
        sharey=True,
        gridspec_kw={'wspace': 0.25}
    )

    for col_idx, p_left in enumerate(p_left_values):

        ax = axes[col_idx]

        # Filter block
        df_block = df_subj[df_subj.p_left == p_left]

        diff_correct = []
        diff_incorrect = []

        for c in contrasts_sorted:

            df_c = df_block[df_block.contrast == c]

            df_c_cor = df_c[df_c.feedback == 1]
            df_c_inc = df_c[df_c.feedback == -1]

            # safe median function (your original)
            def get_med(df, choice):
                vals = np.log10(df[df.choice == choice]['reaction_time'].dropna().values)
                return np.median(vals) if len(vals) > 0 else np.nan

            m_cor_L = get_med(df_c_cor, -1)
            m_cor_R = get_med(df_c_cor,  1)

            m_inc_L = get_med(df_c_inc, -1)
            m_inc_R = get_med(df_c_inc,  1)

            # Left - Right difference
            diff_correct.append(m_cor_L - m_cor_R)
            diff_incorrect.append(m_inc_L - m_inc_R)

        # X positions
        x = np.arange(len(contrasts_sorted))

        # PLOT CORRECT
        ax.plot(
            x, diff_correct,
            marker='o', color=color_correct,
            label='Correct'
        )

        # PLOT INCORRECT
        ax.plot(
            x, diff_incorrect,
            marker='o', color=color_incorrect,
            label='Incorrect'
        )

        # Formatting
        ax.axhline(0, color='black', linewidth=1, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c*100:.0f}' for c in contrasts_sorted])
        ax.set_xlabel("Contrast (%)")

        if col_idx == 0:
            ax.set_ylabel("Δ median RT (log10 s)\nLeft – Right")

        ax.set_title(f"{subj} | Block p_left = {p_left}", fontsize=14, pad=18)

    fig.legend(
        loc='center right',
        bbox_to_anchor=(1.06, 0.5),
        frameon=False,
        fontsize=14
    )

    plt.tight_layout()
    set_plotsize(w=22, h=6)
    plt.show()














































































