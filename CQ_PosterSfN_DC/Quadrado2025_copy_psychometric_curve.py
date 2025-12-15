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
# df_responses = df_responses.query('reaction_time > 0.05').copy()

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


#########################################################################################

#########################################################################################

###### PART 2 - Psychometric ############################################################

#########################################################################################

#########################################################################################


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
trying psychometric curve fit by session 
"""
import psychofit as psy
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from brainbox.behavior.training import plot_psychometric

%matplotlib inline
%pdoc psy

df_responses['contrastLeft'] = np.where(
    (df_responses['signed_contrast'] < 0) |
    ((df_responses['signed_contrast'] == 0) & np.signbit(df_responses['signed_contrast'])),
    df_responses['signed_contrast'],
    np.nan
)

df_responses['contrastRight'] = np.where(
    (df_responses['signed_contrast'] > 0) |
    ((df_responses['signed_contrast'] == 0) & ~np.signbit(df_responses['signed_contrast'])),
    df_responses['signed_contrast'],
    np.nan
)

df_responses['probabilityLeft'] = df_responses.p_left
df_responses['feedbackType'] = df_responses.feedback



# filter
df_responses

df_responses['choice_right'] = (df_responses['choice'] + 1) / 2

df2 = (
    df_responses
    .groupby('signed_contrast')
    .agg(
        ntrials=('choice_right', 'count'),
        fraction=('choice_right', 'mean')
    )
    .reset_index()
)



from brainbox.behavior.training import compute_performance
trials = df_responses
# performance, contrasts, n_contrasts = compute_performance(trials)
# compute performance expressed as probability of choosing right
# performance, contrasts, n_contrasts = compute_performance(trials, prob_right=True)
# compute performance during 0.8 biased block or unbiased
performance, contrasts, n_contrasts = compute_performance(trials, block=0.5)


def compute_n_trials(trials):
    """
    Compute number of trials in trials object

    :param trials: trials object
    :type trials: dict
    returns: int containing number of trials in session
    """
    return trials['choice'].shape[0]


def compute_psychometric(trials, signed_contrast=None, block=None, plotting=False):
    """
    Compute psychometric fit parameters for trials object

    :param trials: trials object that must contain contrastLeft, contrastRight and probabilityLeft
    :type trials: dict
    :param signed_contrast: array of signed contrasts in percent, where -ve values are on the left
    :type signed_contrast: np.array
    :param block: biased block can be either 0.2 or 0.8
    :type block: float
    :return: array of psychometric fit parameters - bias, threshold, lapse high, lapse low
    """

    if signed_contrast is None:
        signed_contrast = trials["signed_contrast"]

    if block is None:
        block_idx = np.full(trials.probabilityLeft.shape, True, dtype=bool)
    else:
        block_idx = trials.probabilityLeft == block

    if not np.any(block_idx):
        return np.nan * np.zeros(4)

    prob_choose_right, contrasts, n_contrasts = compute_performance(trials, signed_contrast=signed_contrast, block=block,
                                                                    prob_right=True)

    if plotting:
        psych, _ = psy.mle_fit_psycho(
            np.vstack([contrasts, n_contrasts, prob_choose_right]),
            P_model='erf_psycho_2gammas',
            parstart=np.array([0., 40., 0.1, 0.1]),
            parmin=np.array([-50., 10., 0., 0.]),
            parmax=np.array([50., 50., 0.2, 0.2]),
            nfits=10)
    else:

        psych, _ = psy.mle_fit_psycho(
            np.vstack([contrasts, n_contrasts, prob_choose_right]),
            P_model='erf_psycho_2gammas',
            parstart=np.array([np.mean(contrasts), 20., 0.05, 0.05]),
            parmin=np.array([np.min(contrasts), 0., 0., 0.]),
            parmax=np.array([np.max(contrasts), 100., 1, 1]))

    return psych

contrasts_2 = [-100. , -25. , 0. , 25. , 100. ]



def plot_psychometric(trials, ax=None, title=None, **kwargs):
    """
    Function to plot pyschometric curve plots a la datajoint webpage
    :param trials:
    :return:
    """

    signed_contrast = trials['signed_contrast']*100
    contrasts_fit = np.arange(-100, 100)

    prob_right_50, contrasts_50, _ = compute_performance(trials, signed_contrast=signed_contrast, block=0.5, prob_right=True)
    pars_50 = compute_psychometric(trials, signed_contrast=signed_contrast, block=0.5, plotting=True)
    prob_right_fit_50 = psy.erf_psycho_2gammas(pars_50, contrasts_fit)

    prob_right_20, contrasts_20, _ = compute_performance(trials, signed_contrast=signed_contrast, block=0.2, prob_right=True)
    pars_20 = compute_psychometric(trials, signed_contrast=signed_contrast, block=0.2, plotting=True)
    prob_right_fit_20 = psy.erf_psycho_2gammas(pars_20, contrasts_fit)

    prob_right_80, contrasts_80, _ = compute_performance(trials, signed_contrast=signed_contrast, block=0.8, prob_right=True)
    pars_80 = compute_psychometric(trials, signed_contrast=signed_contrast, block=0.8, plotting=True)
    prob_right_fit_80 = psy.erf_psycho_2gammas(pars_80, contrasts_fit)

    cmap = ["#E07C12","#320F42","#008F7C"]

    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = plt.gcf()

    # TODO error bars

    fit_50 = ax.plot(contrasts_fit, prob_right_fit_50, color=cmap[1])
    data_50 = ax.scatter(contrasts_50, prob_right_50, color=cmap[1])
    fit_20 = ax.plot(contrasts_fit, prob_right_fit_20, color=cmap[0])
    data_20 = ax.scatter(contrasts_20, prob_right_20, color=cmap[0])
    fit_80 = ax.plot(contrasts_fit, prob_right_fit_80, color=cmap[2])
    data_80 = ax.scatter(contrasts_80, prob_right_80, color=cmap[2])
    ax.legend([fit_50[0], data_50, fit_20[0], data_20, fit_80[0], data_80],
              ['p_L=0.5 fit', 'p_L=0.5 data', 'p_L=0.2 fit', 'p_L=0.2 data', 'p_L=0.8 fit', 'p_L=0.8 data'],
              loc='lower right', frameon=False)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Prob. choosing right')
    ax.set_xlabel('Contrasts')
    
    plt.xticks(contrasts_2)
    plt.axhline(y=0.5,color = 'gray', linestyle = '--',linewidth=0.25) 
    plt.axvline(x=0.5,color = 'gray', linestyle = '--',linewidth=0.25) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if title:
        ax.set_title(title)

    return fig, ax
# fig, ax = plot_psychometric(df_responses)
fig, ax = plot_psychometric(
    df_responses,
    figsize=(6, 6)   # width, height in inches
)










# %%
