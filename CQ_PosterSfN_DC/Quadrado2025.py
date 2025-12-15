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


#### RESPONSE COLLECTION LOOP ##################################################

# Load the sessions
df_sessions = pd.read_parquet(f'/home/kceniabougrova/Documents/NM_project_fromIBLserver/NM_project2/CQ_PosterSfN_DC/{SESSIONS_FNAME}')

# Remove definitely problematic sessions
df_sessions = df_sessions.query('eid not in @EIDS_TO_DROP')

# Restrict the dataframe based on session QC
# ~df_sessions = pd.read_parquet('metadata/sessions_2025-10-31-20h10.pqt')
# ~df_sessions = df_sessions.query('session_status == "good"')
# ~df_sessions['has_photometry'] = df_sessions['alf/photometry/photometry.signal.pqt']
# ~df_sessions = df_sessions.query('has_photometry == True')

# Restrict the dataframe to sessions we're interested in
# ~df_sessions = df_sessions.query('NM == "ACh" and session_type == "biased"')
df_sessions = df_sessions.query('session_type in @SESSION_TYPES')


# Load the insertions table
df_insertions = pd.read_csv('/home/kceniabougrova/Documents/NM_project_fromIBLserver/NM_project2/CQ_PosterSfN_DC/insertions_all.csv')
df_insertions['hemisphere'] = df_insertions['X-ml_um'].apply(lambda x: 'l' if x > 0 else 'r')

# Connect to the database
one = ONE()

# Loop over sessions
responses = []
exceptions_log = []
for idx, session in tqdm(df_sessions.iterrows(), total=len(df_sessions)):
    try:  # try to get data the new way
        loader = PhotometrySessionLoader(eid=session['eid'], one=one)
        loader.load_photometry()
        loader.load_trials()
    except Exception as e:
        exception_info = {  # collect exception info
            'eid': session['eid'],
            'exception_type': type(e).__name__,
            'exception_message': str(e),
            'traceback': traceback.format_exc(),
            'description': "error loading data"
        }
        exceptions_log.append(exception_info)
        pprint(exception_info)
        continue

    # Make sure some basic QC checks are passed
    n_trials = len(loader.trials)
    try:  # check the number of trials in the session
        assert n_trials >= N_TRIALS
    except Exception as e:
        exception_info = {  # collect exception info
            'eid': session['eid'],
            'exception_type': type(e).__name__,
            'exception_message': str(e),
            'traceback': traceback.format_exc(),
            'description': "too few trials in session"
        }
        exceptions_log.append(exception_info)
        pprint(exception_info)
        continue
    full_photometry = pd.concat(
        [loader.photometry['GCaMP'], loader.photometry['Isosbestic']]
    ).sort_index()
    n_early = metrics.n_early_samples(full_photometry)
    try:  # check the signal has no sampling issues
        assert n_early == 0
    except Exception as e:
        exception_info = {  # collect exception info
            'eid': session['eid'],
            'exception_type': type(e).__name__,
            'exception_message': str(e),
            'traceback': traceback.format_exc(),
            'description': "photometry sampling error"
        }
        exceptions_log.append(exception_info)
        pprint(exception_info)
        continue

    # Get the insertion info for this subject
    subject = session['subject']
    insertions = df_insertions.query('subject == @subject')

    # Create a signed contrast column
    loader.trials['contrastLeft'] = -1 * loader.trials['contrastLeft']
    loader.trials['signed_contrast'] = loader.trials['contrastRight'].fillna(
        loader.trials['contrastLeft']
        )

    loader.trials['reaction_times'] = (
        loader.trials['response_times'] - loader.trials['stimOn_times']
        )

    # Loop over target brain areas for this subject
    for target in loader.photometry['GCaMP'].columns:
        photometry = loader.photometry['GCaMP'][target]

        # Check we will be able to say which hemisphere the fiber is in
        try:
            single_insertion = len(insertions) == 1
            hyphenated_target = len(target.split('-')) == 2
            assert single_insertion or hyphenated_target
            if hyphenated_target:
                hemi_in_target = target.split('-')[1] in ['l', 'r']
                assert hemi_in_target
        except Exception as e:
            exception_info = {  # collect exception info
                'eid': session['eid'],
                'exception_type': type(e).__name__,
                'exception_message': str(e),
                'traceback': traceback.format_exc(),
                'description': "implant laterality not specified"
            }
            exceptions_log.append(exception_info)
            pprint(exception_info)
            continue

        n_unique = metrics.n_unique_samples(photometry)
        n_edges = metrics.n_edges(photometry)
        try:
            assert n_unique > 500
        except Exception as e:
            exception_info = {  # collect exception info
                'eid': session['eid'],
                'exception_type': type(e).__name__,
                'exception_message': str(e),
                'traceback': traceback.format_exc(),
                'description': "digitzed signal"
            }
            exceptions_log.append(exception_info)
            pprint(exception_info)
            continue

        # Use the appropriate information to assign laterality to the fiber
        if len(target.split('-')) == 2:  # try target name first
            hemisphere = target.split('-')[1]
        else:  # fall back to insertions table
            hemisphere = insertions['hemisphere'].iloc[0]

        # Get the time points relative to the event (same for all)
        tpts = get_response_tpts(
            loader.photometry['GCaMP'][target]
            )

        # For each event
        for event in EVENTS:

            # Restrict to trials in the photometry time-base
            ## FIXME: check why this happens so often
            trials = loader.trials[
                (loader.trials[event] - 1 >= photometry.index.min()) &
                (loader.trials[event] + 2 <= photometry.index.max())
                ]
            try:
                assert len(trials) == len(loader.trials)
            except Exception as e:
                exception_info = {
                    'eid': session['eid'],
                    'exception_type': type(e).__name__,
                    'exception_message': str(e),
                    'traceback': traceback.format_exc(),
                    'description': "trials outside of photometry time base"
                }
                exceptions_log.append(exception_info)

            # Collect the info in a dict
            resp_dict = {
                'subject': session['subject'],
                'eid': session['eid'],
                'session_type': session['session_type'],
                'NM': session['NM'],
                'target': target.split('-')[0],
                'hemisphere': hemisphere,
                'p_left': trials['probabilityLeft'].values,
                'signed_contrast': trials['signed_contrast'].values,
                'choice': trials['choice'].values,
                'feedback': trials['feedbackType'].values,
                'reaction_time': trials['reaction_times'].values,
                'event': event,
                'tpts': tpts,
                'n_edges': n_edges
                }

            # Get the responses for each trial
            resp_dict['response'] = get_responses(
                photometry, trials, event
                ).T

            # Append to list
            responses.append(resp_dict)

# Convert list to dataframe, and 'explode' such that each trial gets a row
df_responses = pd.DataFrame(responses).explode(
    ['response', 'reaction_time', 'signed_contrast', 'choice', 'p_left', 'feedback']
    ).reset_index(drop=True)
# Convert columns to float
cols_to_fix = ['reaction_time', 'choice', 'signed_contrast', 'p_left', 'feedback']
df_responses[cols_to_fix] = df_responses[cols_to_fix].astype(float)

# Save the response dataframe
timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%M")
RESPONSES_FNAME = f'responses_{timestamp}.pqt'
df_responses.to_parquet(f'responses_{timestamp}.pqt')






#%% 
#### PREPARE DATA FOR ANALYSIS #################################################

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
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Congruent/Incngruent trial based on stimulus side (signed_contrast) and block type (p_left)
def get_congruency(row):
    sc = row['relative_contrast_str']
    p = row['p_left']
    is_left = np.signbit(sc)  # True for -0.0 and negatives, False for +0.0 and positives

    if (is_left and p == 0.8) or (not is_left and p == 0.2):
        return 'congruent'
    elif (is_left and p == 0.2) or (not is_left and p == 0.8):
        return 'incongruent'
    else:
        return 'neutral'

df_responses['congruency'] = df_responses.apply(get_congruency, axis=1)
# ------------------------------------------------------------------


# Log-transform contrast (need to eleminate 0 contrast first)
# ~df_responses['log_contrast'] = df_responses['contrast'].apply(np.log10)

# Normalize the responses
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


#### PLOT RESULTS ##############################################################

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
    orientation='horizontal'
    )
cmap = plt.colormaps['Greys']
rt_colors = cmap(np.linspace(0.4, 0.99, len(rts)))
for pc, color in zip(parts['bodies'], rt_colors):
    pc.set_facecolor(color)
parts['cmedians'].set_colors(rt_colors)
for i, rt in enumerate(rts):
    median = np.median(rt)
    ax.text(
        median, i + 1, f'{10**median:.2f}s', rotation=45, ha='left', va='bottom'
        )
contrasts = df_responses['contrast'].unique()
ax.set_yticks(np.arange(1, len(contrasts) + 1))
ax.set_yticklabels([f'{c*100:.0f}' for c in sorted(contrasts)])
ax.set_ylabel('Contrast level')
xticks = np.linspace(-2, 2, 6)
ax.set_xticks(xticks)
ax.set_xticklabels(['$10^{%d}$' % t for t in xticks])
ax.set_xlim([-2.1, 2])
ax.set_xlabel('Reaction time (s)')
clip_axes_to_ticks(ax=ax)
set_plotsize(w=12, h=4, ax=ax)
fig.savefig('figures/reaction_times.svg')


CONGRUENCY_COLORS = {
    ('congruent', 1): '#1b7837',    # dark green – correct congruent
    ('incongruent', 1): '#a6dba0',  # light green – correct incongruent
    ('congruent', 0): '#b2182b',    # dark red – incorrect congruent
    ('incongruent', 0): '#ef8a62',  # orange-red – incorrect incongruent
}

# Plot responses by contrast for each target-NM in the 50-50 block
# df_unbiased = df_responses.query('p_left == 0.5').copy()
df_biased = df_responses.query('p_left in [0.2, 0.8]').copy()
# for (target, NM), df_target in df_unbiased.groupby(['target', 'NM']):
for (target, NM), df_target in df_biased.groupby(['target', 'NM']):
    for event in EVENTS:
        event_name = event.split("_")[0]
        df_event = df_target.query('event == @event')
        n_sessions = df_target['eid'].nunique()
        n_mice = df_target['subject'].nunique()
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(
            f'{target}-{NM} - {event_name} ({n_sessions} sessions, {n_mice} mice)'
            )
        for ax, feedback in zip(axs, [1, -1]):
            label = 'Correct' if feedback == 1 else 'Incorrect'
            ax.set_title(f'{label} trials')
            trial_groups = df_event.query(f'feedback == {feedback}').groupby('contrast')
            colors = NM_CMAPS[NM](np.linspace(0.3, 0.99, len(trial_groups)))
            for (contrast, trials), color in zip(trial_groups, colors):
                plot_mean_response(
                    trials,
                    ax=ax,
                    # ~color=CONTRAST_COLORS[f'contrast_{contrast}'],
                    color=color,
                    label=f'{contrast * 100:.0f} (N = {len(trials)})'
                    )
            ax.axhline(0, ls='--', color='gray')
            ax.axvline(0, ls='--', color='gray')
            ax.axvspan(*RESPONSE_WINDOW, color='gray', alpha=0.15)
            ax.set_xticks(np.linspace(PSTH_WINDOW[0], PSTH_WINDOW[1], 3))
            ax.set_xlabel('Time from event (s)')
            ax.set_ylabel('$\Delta$F / F')
            ax.legend(
                title='Contrast', frameon=False, loc='upper left', bbox_to_anchor=(1, 1)
                )
        ymax = max([ax.get_yticks().max() for ax in axs])
        ymin = min([ax.get_yticks().min() for ax in axs])
        for ax in axs:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
            ax.set_yticks(np.linspace(ymin, ymax, 5))
            clip_axes_to_ticks(ax=ax)
        set_plotsize(w=34, h=12, ax=ax)
        fig.tight_layout()
        fig.savefig(f'figures/{target}-{NM}_{event_name}.svg')

# Plot response magnitude for signed contrasts in the 50-50 block
# df_unbiased = df_responses.query('p_left == 0.5').copy()
df_biased = df_responses.query('p_left in [0.2, 0.8]').copy()
response_window = '-'.join([str(t) for t in RESPONSE_WINDOW])
plot_subjects = False
# for (target, NM), df_target in df_unbiased.groupby(['target', 'NM']):
for (target, NM), df_target in df_biased.groupby(['target', 'NM']):
    for event in EVENTS:
        event_name = event.split("_")[0]
        df_event = df_target.query('event == @event').copy()

        # Center the means for each subject, then add the grand mean
        grand_mean = df_event['response_mean'].mean()
        df_event['centered_mean'] = df_event.groupby('subject')['response_mean'].transform(
            lambda x: x - x.mean()
            ) + grand_mean

        fig, axs = plt.subplots(1, 2, gridspec_kw={'wspace':0.05}, sharey=True)
        fig.suptitle(f'{target}-{NM} - {event_name} ({response_window}s)')
        for ax, side in zip(axs, [True, False]):
            df_side = df_event.query('side == @side')
            contrasts = sorted(df_side['relative_contrast'].unique())

            # Plot individual subjects first (thin lines, no errorbars)
            if plot_subjects:
                for subject in df_side['subject'].unique():
                    df_subj = df_side.query('subject == @subject')
                    for feedback, linestyle in zip([1, -1], ['-', '--']):
                        contrast_groups = df_subj.query('feedback == @feedback').groupby('relative_contrast')
                        sorted_groups = [
                            contrast_groups.get_group(c)
                            if c in contrast_groups.groups else []
                            for c in contrasts
                            ]
                        ax.plot(
                            np.arange(len(contrasts)),
                            [group['centered_mean'].mean() if len(group) > 5 else np.nan for group in sorted_groups],
                            color=NM_COLORS[NM],
                            alpha=0.3,
                            linewidth=1,
                            linestyle=linestyle
                        )

            for feedback, linestyle in zip([1, -1], ['-', '--']):
                contrast_groups = df_side.query('feedback == @feedback').groupby('relative_contrast')
                sorted_groups = [contrast_groups.get_group(c) for c in contrasts]
                ax.errorbar(
                    np.arange(len(contrasts)),
                    [group['centered_mean'].mean() if len(group) > 10 else np.nan for group in sorted_groups],
                    yerr=[group['centered_mean'].sem() if len(group) > 10 else np.nan for group in sorted_groups],
                    marker='o',
                    color=NM_COLORS[NM],
                    linestyle=linestyle,
                    label=feedback
                    )

            if side:
                ax.text(0.05, 0.05, 'Contra', ha='left', va='bottom', transform=ax.transAxes)
            else:
                ax.text(0.95, 0.05, 'Ipsi', ha='right', va='bottom', transform=ax.transAxes)

            ax.set_xticks(np.arange(len(contrasts)))
            ax.set_xticklabels([f'{c*100:.0f}' for c in contrasts])
            ax.set_xlabel('Contrast level')
            ax.axhline(0, ls='--', color='gray')
            if side:
                ax.set_ylabel('$\Delta$F/F')
            else:
                ax.yaxis.set_visible(False)
                ax.tick_params(left=False)
                ax.spines['left'].set_visible(False)
                ax.legend(title='Reward', loc='upper left', bbox_to_anchor=(1, 1))

        if plot_subjects:
            # Calculate y-limits from subject means
            subject_means = df_event.groupby(
                ['subject', 'relative_contrast', 'feedback', 'side']
                )['centered_mean'].mean()
            y_min = subject_means.min()
            y_max = subject_means.max()
        else:
            # Calculate y-limits from condition means
            condition_means = df_event.groupby(
                ['relative_contrast', 'feedback', 'side']
                )['centered_mean'].mean()
            y_min = condition_means.min()
            y_max = condition_means.max()
        for ax in axs:
            ax.set_yticks(nice_ticks(y_min, y_max, d=3, n_ticks=3))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            clip_axes_to_ticks(ax=ax)

        set_plotsize(w=18, h=12, ax=ax)

        fname = f'{target}-{NM}_{event_name}_mean{response_window}s'
        if plot_subjects:
            fname += '_subjects'
        fig.savefig('figures/' + fname + '.svg')


#### STATISTICAL ANALYSIS ######################################################
# Note: This requires the pymer4 package (depends on R), but could also be done
# with statsmodels (pure python)

from pymer4.models import Lmer
from rpy2.robjects import pandas2ri
pandas2ri.activate()

df_unbiased = df_responses.query('p_left == 0.5').copy()
response_window = '-'.join([str(t) for t in RESPONSE_WINDOW])
lmm_formula = f'response_mean ~ contrast * side * feedback + (1 | subject)'
for (target, NM), df_target in df_unbiased.groupby(['target', 'NM']):
    if df_target['subject'].nunique() < 2:
        continue
    for event in ['stimOn_times', 'feedback_times']:
        event_name = event.split('_')[0]
        df_event = df_target.query('event == @event').copy().reset_index(drop=True)
        print("\n=================================================================")
        print(f'{target}-{NM} -- {event_name}')
        print(f'LMM: {lmm_formula}')
        print("=================================================================")

        df_event['side'] = df_event['side'].apply(
            lambda x: 'Contra' if x else 'Ipsi'
        ).astype('category')
        df_event['feedback'] = df_event['feedback'].apply(lambda x: str(x)).astype('category')
        df_event['subject'] = df_event['subject']
        cols = ['subject', 'side', 'contrast', 'feedback', 'response_mean']

        model = Lmer(lmm_formula, data=df_event[cols])
        result = model.fit()
        print(model.warnings)
        print(result)
        df_result = result.copy()
        df_result['formula'] = lmm_formula
        df_result.to_csv(f'results/{target}-{NM}_{event_name}_mean{response_window}s.csv')


#### DEBUGGING #################################################################

def rvr(group):
    r = np.stack(group['response'])
    return r.mean(axis=0).std() / r.std(axis=0).mean()





#%%

# ================================================================
# Plot response magnitude for congruent vs incongruent (0.2 & 0.8 blocks and L/R contrast sides)
# ================================================================

df_biased = df_responses.query('p_left in [0.2, 0.8]').copy()
response_window = '-'.join([str(t) for t in RESPONSE_WINDOW])
plot_subjects = False  # toggle for thin subject lines

df_biased['relative_contrast_str'] = df_biased['relative_contrast'].apply(
    lambda x: (
        "-0" if np.signbit(x) and x == 0
        else "0" if not np.signbit(x) and x == 0
        else f"{x*100:.2f}"  # express as percentage
    )
)

# Define desired label order (matches IBL contrast set)
desired_order = ['-100', '-25', '-12.50', '-6.25', '-0', '0',
                 '6.25', '12.50', '25', '100']

# Colors for congruency × feedback
CONGRUENCY_COLORS = {
    ('congruent', 1): '#1b7837',    # dark green
    ('incongruent', 1): '#a6dba0',  # light green
    ('congruent', -1): '#b2182b',   # dark red
    ('incongruent', -1): '#ef8a62', # orange-red
}

for (target, NM), df_target in df_biased.groupby(['target', 'NM']):
    for event in EVENTS:
        event_name = event.split("_")[0]
        df_event = df_target.query('event == @event').copy()

        # Center per subject
        grand_mean = df_event['response_mean'].mean()
        df_event['centered_mean'] = df_event.groupby('subject')['response_mean'].transform(
            lambda x: x - x.mean()
        ) + grand_mean

        # Keep only contrasts present in data, ordered per desired list
        # contrasts = [c for c in desired_order if c in df_event['relative_contrast_str'].unique()]
        # Sort contrasts in intuitive signed order, preserving -0 and 0 distinction
        contrast_order = ['-100.00', '-25.00', '-12.50', '-6.25', '-0', '0', '6.25', '12.50', '25.00', '100.00']
        contrasts = [c for c in contrast_order if c in df_event['relative_contrast_str'].unique()]


        fig, ax = plt.subplots(figsize=(13, 8))
        fig.suptitle(f'{target}-{NM} - {event_name} ({response_window}s)')

        # Main plot: mean ± SEM (with n-trials in legend)
        for (congruency, feedback), linestyle in zip(
            [('congruent', 1), ('incongruent', 1), ('congruent', -1), ('incongruent', -1)],
            ['-', '-', '--', '--']
        ):
            df_cond = df_event.query('congruency == @congruency and feedback == @feedback')
            grouped = df_cond.groupby('relative_contrast_str')['centered_mean']
            means = [grouped.mean().get(lbl, np.nan) for lbl in contrasts]
            sems  = [grouped.sem().get(lbl, np.nan) for lbl in contrasts]

            # 🧮 Count number of trials for this condition
            n_trials = len(df_cond)
            label = f'{congruency.capitalize()} - {"Correct" if feedback == 1 else "Incorrect"} (n={n_trials})'

            # 📈 Plot
            ax.errorbar(
                np.arange(len(contrasts)),
                means,
                yerr=sems,
                marker='o',
                color=CONGRUENCY_COLORS.get((congruency, feedback), 'gray'),
                linestyle=linestyle,
                label=label,
                linewidth=2,
                alpha=0.9
            )


        # --- Aesthetics ---
        ax.axhline(0, ls='--', color='gray')
        ax.set_xticks(np.arange(len(contrasts)))
        ax.set_xticklabels(contrasts)
        ax.set_xlabel('Contrast level (%)')
        ax.set_ylabel('$\Delta$F/F')
        ax.legend(title='Condition', frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Dynamic y-axis scaling
        condition_means = df_event.groupby(
            ['relative_contrast_str', 'congruency', 'feedback']
        )['centered_mean'].mean()
        y_min, y_max = condition_means.min(), condition_means.max()
        ax.set_yticks(np.linspace(y_min, y_max, 3))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # Save
        fname = f'{target}-{NM}_{event_name}_congruency_{response_window}s'
        fig.savefig('figures/' + fname + '.svg', bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

CONGRUENCY_COLORS = {
    ('congruent', 1): 'darkgreen',
    ('incongruent', 1): 'lightgreen',
    ('congruent', -1): 'darkred',
    ('incongruent', -1): 'orange'
}

for (target, NM), df_target in df_biased.groupby(['target', 'NM']):
    for event in EVENTS:
        event_name = event.split("_")[0]
        df_event = df_target.query('event == @event').copy()

        # Center per subject
        grand_mean = df_event['response_mean'].mean()
        df_event['centered_mean'] = df_event.groupby('subject')['response_mean'].transform(
            lambda x: x - x.mean()
        ) + grand_mean

        # --- Define contrast order ---
        contrast_order = ['-100.00', '-25.00', '-12.50', '-6.25', '-0', '0', '6.25', '12.50', '25.00', '100.00']
        contrasts = [c for c in contrast_order if c in df_event['relative_contrast_str'].unique()]

        # --- Split the x-axis between '-0' and '0' ---
        split_point = contrasts.index('-0') + 1 if '-0' in contrasts else len(contrasts)//2
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13, 8), sharey=True,
            gridspec_kw={'width_ratios': [split_point, len(contrasts) - split_point]}
        )
        fig.suptitle(f'{target}-{NM} - {event_name} ({response_window}s)')

        # --- Plot each condition ---
        for (congruency, feedback), linestyle in zip(
            [('congruent', 1), ('incongruent', 1), ('congruent', -1), ('incongruent', -1)],
            ['-', '-', '--', '--']
        ):
            df_cond = df_event.query('congruency == @congruency and feedback == @feedback')
            grouped = df_cond.groupby('relative_contrast_str')['centered_mean']
            means = [grouped.mean().get(lbl, np.nan) for lbl in contrasts]
            sems  = [grouped.sem().get(lbl, np.nan) for lbl in contrasts]
            n_trials = len(df_cond)
            label = f'{congruency.capitalize()} - {"Correct" if feedback == 1 else "Incorrect"} (n={n_trials})'
            color = CONGRUENCY_COLORS.get((congruency, feedback), 'gray')

            # Left side (negative contrasts)
            ax1.errorbar(
                np.arange(split_point),
                means[:split_point],
                yerr=sems[:split_point],
                marker='o', color=color,
                linestyle=linestyle, label=label,
                linewidth=2, alpha=0.9
            )

            # Right side (positive contrasts)
            ax2.errorbar(
                np.arange(split_point, len(contrasts)),
                means[split_point:],
                yerr=sems[split_point:],
                marker='o', color=color,
                linestyle=linestyle,
                linewidth=2, alpha=0.9
            )

        # ---------
        for ax in (ax1, ax2):
            ax.axhline(0, ls='--', color='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # --- Hide y-axis on the right panel only ---
        ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax2.spines['left'].set_visible(False)

        ax1.set_ylabel('$\Delta$F/F')
        condition_means = df_event.groupby(
            ['relative_contrast_str', 'congruency', 'feedback']
        )['centered_mean'].mean()
        y_min, y_max = condition_means.min(), condition_means.max()
        ax1.set_yticks(np.linspace(y_min, y_max, 3))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))


        ax1.set_xticks(np.arange(split_point))
        ax1.set_xticklabels(contrasts[:split_point])
        ax1.set_xlabel('Contrast level (%)')

        ax2.set_xticks(np.arange(split_point, len(contrasts)))
        ax2.set_xticklabels(contrasts[split_point:])
        ax2.set_xlabel('Contrast level (%)')


        # --- Layout & legend ---
        plt.subplots_adjust(wspace=0.05)
        ax2.legend(title='Condition', frameon=False, loc='upper left', bbox_to_anchor=(1, 1))

        # --- Save & show ---
        fname = f'{target}-{NM}_{event_name}_congruency_{response_window}s'
        # fig.savefig('figures/broken0_' + fname + '.svg', bbox_inches='tight') 
        fig.savefig('figures/broken0_' + fname + '.png', bbox_inches='tight')

        plt.tight_layout()
        plt.show()
        plt.close(fig)

#%% 
# ===============================================================
# HEATMAPS

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ================================================================
# 1️⃣ Prepare base dataframe
# ================================================================
df = df_biased.copy()
df = df.query("p_left in [0.2, 0.8]").copy()

# Center per subject × event (ΔF/F baseline correction)
df['centered_mean'] = (
    df.groupby(['subject', 'event'])['response_mean']
      .transform(lambda x: x - x.mean()) + df['response_mean'].mean()
)

# Combine congruency and hemisphere for y-axis grouping
df['condition'] = df['congruency'].str.capitalize() + " – " + df['hemisphere'].astype(str)

# ================================================================
# 2️⃣ Define plotting parameters
# ================================================================
contrast_order = ['-100.00', '-25.00', '-12.50', '-6.25', '-0', '0',
                  '6.25', '12.50', '25.00', '100.00']

nm_order = ['DA', '5HT', 'NE', 'ACh']
event_order = ['stimOn_times', 'feedback_times']

# Define NM colors
nm_colors = {
    'DA': "#ae0900",    # red
    '5HT': "#660178",   # purple
    'NE': "#0261a0",    # blue
    'ACh': "#098a00"    # green
}

# Create colormaps per NM
def make_cmap(base_color):
    return LinearSegmentedColormap.from_list("", ["#ffffff", base_color])

nm_cmaps = {nm: make_cmap(col) for nm, col in nm_colors.items()}

# ================================================================
# 3️⃣ Aggregate data per NM × condition × contrast
# ================================================================
df_mean = (
    df.groupby(['NM', 'event', 'condition', 'relative_contrast_str'], as_index=False)
      .agg(centered_mean=('centered_mean', 'mean'))
)

# Order conditions logically
cond_order = sorted(df_mean['condition'].unique(), key=lambda x: (x.split(" – ")[0], x.split(" – ")[1]))

# ================================================================
# 4️⃣ Plot per event (one figure with 4 NM heatmaps)
# ================================================================
for event in event_order:
    df_event = df_mean.query("event == @event").copy()
    fig, axes = plt.subplots(len(nm_order), 1, figsize=(9, 8), sharex=True)
    fig.suptitle(f"ΔF/F by Contrast and Condition – {event.replace('_times','')}",
                 fontsize=14, weight='bold')

    for i, NM in enumerate(nm_order):
        ax = axes[i]
        df_plot = df_event.query("NM == @NM").copy()

        # Pivot table (rows = condition, cols = contrast)
        pivot = df_plot.pivot_table(
            index="condition", columns="relative_contrast_str", values="centered_mean"
        )
        pivot = pivot.reindex(columns=contrast_order)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=nm_cmaps[NM],
            center=0,
            linewidths=0.4,
            linecolor="gray",
            cbar=True,
            cbar_kws={'label': 'ΔF/F'},
            vmin=df_mean['centered_mean'].quantile(0.05),
            vmax=df_mean['centered_mean'].quantile(0.95)
        )

        ax.set_title(NM, color=nm_colors[NM], fontsize=13, weight='bold')
        ax.set_ylabel("Congruency – Hemisphere")
        ax.set_xlabel("") if i < len(nm_order)-1 else ax.set_xlabel("Contrast level (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



































# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ==========================================================
# 8-GROUP LABELS (human-readable + symbol)
# ==========================================================

def extract_stim_side(row):
    sc = row["signed_contrast"]
    is_left = np.signbit(float(sc))
    return "L" if is_left else "R"

def extract_hemi_side(row):
    return "L" if row["hemisphere"] == -1 else "R"

def extract_block_side(row):
    if row["p_left"] == 0.8:
        return "LeftBias(0.8)"
    elif row["p_left"] == 0.2:
        return "RightBias(0.2)"
    return "Neutral"

def get_group8_symbol(row):
    stim = row["stim_side"]
    hemi = row["hemi_side"]
    p = row["p_left"]

    if p == 0.8:
        if stim=="L" and hemi=="L": return "80_L_L"
        if stim=="L" and hemi=="R": return "80_L_R"
        if stim=="R" and hemi=="L": return "80_R_L"
        if stim=="R" and hemi=="R": return "80_R_R"
    elif p == 0.2:
        if stim=="L" and hemi=="L": return "20_L_L"
        if stim=="L" and hemi=="R": return "20_L_R"
        if stim=="R" and hemi=="L": return "20_R_L"
        if stim=="R" and hemi=="R": return "20_R_R"
    return None

def get_group8_label(row):
    # readable string
    return (
        f"{row['block_side']} | "
        f"Stim:{row['stim_side']} | "
        f"Hemi:{row['hemi_side']}"
    ) 

df_biased = df_responses.query('p_left in [0.2, 0.8]').copy()


# Add columns
df_biased["stim_side"]  = df_biased.apply(extract_stim_side, axis=1)
df_biased["hemi_side"]  = df_biased.apply(extract_hemi_side, axis=1)
df_biased["block_side"] = df_biased.apply(extract_block_side, axis=1)

df_biased["group8_symbol"]  = df_biased.apply(get_group8_symbol, axis=1)
df_biased["group8_readable"] = df_biased.apply(get_group8_label, axis=1)









# One color per 8-group symbol
#cold blue = congruent LEFT
    #ipsi contra ignore
#warm red = congruent RIGHT
    #ipsi contra ignore
#cold green = inc LEFT
    #ipsi contra ignore
#warm yellowish = inc RIGHT
    #ipsi contra ignore



GROUP8_COLORS = {
    "80_L_L": "#0096c7", #ipsi
    "80_L_R": "#0096c7", #c
    "80_R_L": "#ffba08", #c
    "80_R_R": "#ffba08", #ipsi
    "20_L_L": "#55a630", #ipsi
    "20_L_R": "#55a630", #c
    "20_R_L": "#c77dff", #c
    "20_R_R": "#c77dff", #ipsi
}

for (target, NM), df_target in df_biased.groupby(['target', 'NM']):
    for event in EVENTS:
        event_name = event.split("_")[0]
        df_event = df_target.query('event == @event').copy()

        # Center per subject
        grand_mean = df_event['response_mean'].mean()
        df_event['centered_mean'] = df_event.groupby('subject')['response_mean'].transform(
            lambda x: x - x.mean()
        ) + grand_mean

        # --- Define contrast order ---
        contrast_order = ['-100.00', '-25.00', '-12.50', '-6.25', '-0', '0',
                          '6.25', '12.50', '25.00', '100.00']
        contrasts = [c for c in contrast_order
                     if c in df_event['relative_contrast_str'].unique()]


        # --- Split the x-axis between '-0' and '0' ---
        split_point = contrasts.index('-0') + 1 if '-0' in contrasts else len(contrasts)//2
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13, 8), sharey=True,
            gridspec_kw={'width_ratios': [split_point,
                                          max(len(contrasts) - split_point, 1)]}
        )
        fig.suptitle(f'{target}-{NM} - {event_name} ({response_window}s)')

        # --- Plot each 8-group × feedback ---
        for group8, df_g in df_event.groupby('group8_symbol'):
            if group8 is None:
                continue
            color = GROUP8_COLORS.get(group8, 'black')

            for feedback, linestyle in zip([1, -1], ['-', '--']):
                df_cond = df_g.query('feedback == @feedback')
                if df_cond.empty:
                    continue

                grouped = df_cond.groupby('relative_contrast_str')['centered_mean']
                means = [grouped.mean().get(lbl, np.nan) for lbl in contrasts]
                sems  = [grouped.sem().get(lbl, np.nan) for lbl in contrasts]
                n_trials = len(df_cond)

                label = (f'{group8} - '
                         f'{"Correct" if feedback == 1 else "Incorrect"} '
                         f'(n={n_trials})')

                # Left side (negative & -0)
                ax1.errorbar(
                    np.arange(split_point),
                    means[:split_point],
                    yerr=sems[:split_point],
                    marker='o', color=color,
                    linestyle=linestyle, label=label,
                    linewidth=2, alpha=0.9
                )

                # Right side (0 & positives)
                if split_point < len(contrasts):
                    ax2.errorbar(
                        np.arange(split_point, len(contrasts)),
                        means[split_point:],
                        yerr=sems[split_point:],
                        marker='o', color=color,
                        linestyle=linestyle,
                        linewidth=2, alpha=0.9
                    )

        # --------- common formatting ----------
        for ax in (ax1, ax2):
            ax.axhline(0, ls='--', color='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Hide y-axis on the right panel only
        ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax2.spines['left'].set_visible(False)

        ax1.set_ylabel('$\Delta$F/F')

        # y-limits based on group8
        condition_means = df_event.groupby(
            ['relative_contrast_str', 'group8_symbol', 'feedback']
        )['centered_mean'].mean()
        y_min, y_max = condition_means.min(), condition_means.max()
        ax1.set_yticks(np.linspace(y_min, y_max, 3))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax1.set_xticks(np.arange(split_point))
        ax1.set_xticklabels(contrasts[:split_point])
        ax1.set_xlabel('Contrast level (%)')

        if split_point < len(contrasts):
            ax2.set_xticks(np.arange(split_point, len(contrasts)))
            ax2.set_xticklabels(contrasts[split_point:])
        ax2.set_xlabel('Contrast level (%)')

        # Layout & legend
        plt.subplots_adjust(wspace=0.05)
        ax2.legend(title='Group (p_left × stim × hemi × feedback)',
                   frameon=False, loc='upper left', bbox_to_anchor=(1, 1))

        fname = f'{target}-{NM}_{event_name}_group8_{response_window}s'
        # fig.savefig('figures/CONG_INCONG_I_C_broken0_' + fname + '.png', bbox_inches='tight')
        # fig.savefig('figures/CONG_INCONG_I_C_broken0_' + fname + '.svg', bbox_inches='tight')

        plt.tight_layout()
        plt.show()
        plt.close(fig)

# %%
