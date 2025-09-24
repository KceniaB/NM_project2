#%% 
""" 
KB 2025-09-23 
VERSION3 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

from one.api import ONE #always after the imports
one = ONE() 

"""
CHANGE HERE THE EVENT AND THE EID
FOR THE SAME EID YOU CAN CHANGE THE EVENT LATER IN THIS CODE 
"""

eid = '1be2e1e9-e4f2-41ad-9e34-ae27967d41ac'
EVENT = "feedback_times"  # "stimOnTrigger_times" etc


#%%

""" FUNCTIONS """

def load_trials_updated(eid): 
    trials = one.load_object(eid, 'trials')
    ref = one.eid2ref(eid)
    subject = ref.subject
    session_date = str(ref.date) 
    if len(trials['intervals'].shape) == 2: 
        trials['intervals_0'] = trials['intervals'][:, 0]
        trials['intervals_1'] = trials['intervals'][:, 1]
        del trials['intervals']  # Remove original nested array 
    df_trials = pd.DataFrame(trials) 
    idx = 2
    new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
    df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
    # create allSContrasts 
    df_trials['allSContrasts'] = df_trials['allContrasts']
    df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
    df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))
    df_trials[["subject", "date", "eid"]] = [subject, session_date, eid]    
    df_trials["reactionTime"] = df_trials["firstMovement_times"] - df_trials["stimOnTrigger_times"]
    df_trials["responseTime"] = df_trials["response_times"] - df_trials["stimOnTrigger_times"] 
    df_trials["quiescenceTime"] = df_trials["stimOnTrigger_times"] - df_trials["intervals_0"] 
    df_trials["trialTime"] = df_trials["intervals_1"] - df_trials["intervals_0"]  

    try: 
        dataset_task_settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json')  
        values = dataset_task_settings.get('LEN_BLOCKS', 'Key not found') 
        # values gives the block length 
        # example for eid = 'be3208c9-43de-44dc-bdc6-ff8963464f98'
        # [90, 27, 82, 50, 30, 30, 31, 78, 64, 83, 24, 42, 74, 72, 34, 41, 52, 56, 68, 39, 45, 88, 37, 35, 29, 69, 85, 52, 37, 78, 80, 28, 68, 95, 34, 36, 42] 

        values_sum = np.cumsum(values) 

        # Initialize a new column 'probL' with NaN values
        df_trials['probL'] = np.nan

        # Set the first block (first `values_sum[0]` rows) to 0.5
        df_trials.loc[:values_sum[0]-1, 'probL'] = 0.5 


        df_trials.loc[values_sum[0]:values_sum[1]-1, 'probL'] = df_trials.loc[values_sum[0], 'probabilityLeft']

        previous_value = df_trials.loc[values_sum[1]-1, 'probabilityLeft'] 


        # Iterate over the blocks starting from values_sum[1]
        for i in range(1, len(values_sum)-1):
            print("i = ", i)
            start_idx = values_sum[i]
            end_idx = values_sum[i+1]-1
            print("start and end _idx = ", start_idx, end_idx)
            
            # Assign the block value based on the previous one
            if previous_value == 0.2:
                current_value = 0.8
            else:
                current_value = 0.2
            print("current value = ", current_value)


            # Set the 'probL' values for the current block
            df_trials.loc[start_idx:end_idx, 'probL'] = current_value
            
            # Update the previous_value for the next block
            previous_value = current_value

        # Handle any remaining rows after the last value_sum block
        if len(df_trials) > values_sum[-1]:
            df_trials.loc[values_sum[-1] + 1:, 'probL'] = previous_value

        # plt.plot(df_trials.probabilityLeft, alpha=0.5)
        # plt.plot(df_trials.probL, alpha=0.5)
        # plt.title(f'behavior_{subject}_{session_date}_{eid}')
        # plt.show() 
    except: 
        pass 

    df_trials["trialNumber"] = range(1, len(df_trials) + 1) 
    return df_trials, subject, session_date 

def verify_length(df_nph): 
    """
    Checking if the length is different
    x = df_470
    y = df_415
    """ 
    x = df_nph[df_nph.wavelength==470.0]
    y = df_nph[df_nph.wavelength==415.0] 
    if len(x) == len(y): 
        print("Option 1: same length :)")
    else: 
        print("Option 2: SOMETHING IS WRONG! Different len's") 
    print("470 = ",x.wavelength.count()," 415 = ",y.wavelength.count())
    return(x,y)


def get_zdFF(reference,signal,smooth_win=10,remove=200,lambd=5e4,porder=1,itermax=50): 
  '''
  Calculates z-score dF/F signal based on fiber photometry calcium-idependent 
  and calcium-dependent signals
  
  Input
      reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
      signal: calcium-dependent signal (usually 465-490 nm excitation for 
                   green fluorescent proteins, or ~560 nm for red), 1D array
      smooth_win: window for moving average smooth, integer
      remove: the beginning of the traces with a big slope one would like to remove, integer
      Inputs for airPLS:
      lambd: parameter that can be adjusted by user. The larger lambda is,  
              the smoother the resulting background, z
      porder: adaptive iteratively reweighted penalized least squares for baseline fitting
      itermax: maximum iteration times
  Output
      zdFF - z-score dF/F, 1D numpy array
  '''
  
  import numpy as np
  from sklearn.linear_model import Lasso

 # Smooth signal
  reference = smooth_signal(reference, smooth_win)
  signal = smooth_signal(signal, smooth_win)
  
 # Remove slope using airPLS algorithm
  r_base=airPLS(reference,lambda_=lambd,porder=porder,itermax=itermax)
  s_base=airPLS(signal,lambda_=lambd,porder=porder,itermax=itermax) 

 # Remove baseline and the begining of recording
  reference = (reference[remove:] - r_base[remove:])
  signal = (signal[remove:] - s_base[remove:])   

 # Standardize signals    
  reference = (reference - np.median(reference)) / np.std(reference)
  signal = (signal - np.median(signal)) / np.std(signal)
  
 # Align reference signal to calcium signal using non-negative robust linear regression
  lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
              positive=True, random_state=9999, selection='random')
  n = len(reference)
  lin.fit(reference.reshape(n,1), signal.reshape(n,1))
  reference = lin.predict(reference.reshape(n,1)).reshape(n,)

 # z dFF    
  zdFF = (signal - reference)
 
  return zdFF


def smooth_signal(x,window_len=10,window='flat'):

    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                'flat' window will produce a moving average smoothing.

    output:
        the smoothed signal        
    """

    import numpy as np

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': # Moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[(int(window_len/2)-1):-int(window_len/2)]


'''
airPLS.py Copyright 2014 Renato Lombardo - renato.lombardo@unipa.it
Baseline correction using adaptive iteratively reweighted penalized least squares

This program is a translation in python of the R source code of airPLS version 2.0
by Yizeng Liang and Zhang Zhimin - https://code.google.com/p/airpls

Reference:
Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive iteratively 
reweighted penalized least squares. Analyst 135 (5), 1138-1146 (2010).

Description from the original documentation:
Baseline drift always blurs or even swamps signals and deteriorates analytical 
results, particularly in multivariate analysis.  It is necessary to correct baseline 
drift to perform further data analysis. Simple or modified polynomial fitting has 
been found to be effective in some extent. However, this method requires user 
intervention and prone to variability especially in low signal-to-noise ratio 
environments. The proposed adaptive iteratively reweighted Penalized Least Squares
(airPLS) algorithm doesn't require any user intervention and prior information, 
such as detected peaks. It iteratively changes weights of sum squares errors (SSE) 
between the fitted baseline and original signals, and the weights of SSE are obtained 
adaptively using between previously fitted baseline and original signals. This 
baseline estimator is general, fast and flexible in fitting baseline.


LICENCE
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''

from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                 the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z


def psth(calcium, times, t_events, fs, peri_event_window):
    """
    Extract peri-event calcium signal aligned to behavioral events.

    Parameters
    ----------
    calcium : np.ndarray
        Photometry signal.
    times : np.ndarray
        Timestamps corresponding to the calcium signal (same clock as t_events).
    t_events : np.ndarray
        Event timestamps to align to (e.g. feedback_times).
    fs : float
        Sampling frequency in Hz.
    peri_event_window : list or tuple
        Time window around event [start, end] in seconds (e.g. [-1, 2]).

    Returns
    -------
    psth_array : np.ndarray
        Array of shape [timepoints, n_trials], each column is a peri-event trace.
    idx_matrix : np.ndarray
        Index matrix of shape [timepoints, n_trials] showing the signal indices used.
    """
    n_trials = len(t_events)
    time_window = np.arange(peri_event_window[0], peri_event_window[1], 1/fs)
    n_timepoints = len(time_window)

    idx_matrix = np.zeros((n_timepoints, n_trials), dtype=int)
    psth_array = np.full((n_timepoints, n_trials), np.nan)

    for i, t_event in enumerate(t_events):
        center_idx = np.searchsorted(times, t_event)

        start_idx = center_idx + int(peri_event_window[0] * fs)
        end_idx = start_idx + n_timepoints

        if start_idx < 0 or end_idx > len(calcium):
            continue  # skip out-of-bounds trials

        idx_range = np.arange(start_idx, end_idx)
        psth_array[:, i] = calcium[idx_range]
        idx_matrix[:, i] = idx_range

    return psth_array, idx_matrix





#%%
#===========================================================================
#                            Pick data from table 
#===========================================================================
"""
search for those 3 files: 
    - _neurophotometrics_fpData_raw.pqt
    - photometryROI_locations.pqt - or just know the region G from which you're recording from 
    - photometry_signal.pqt
# """
# # table_path = '/home/kceniabougrova/Downloads/photometry_ZFM-09139_2025-08-05/_neurophotometrics_fpData_raw.pqt' 
# # photometry_raw_table = pd.read_parquet(table_path)
# table_path = '/home/kceniabougrova/Downloads/photometry_ZFM-09139_2025-08-05/photometryROI_locations.pqt' 
# region_map = pd.read_parquet(table_path)
# table_path = '/home/kceniabougrova/Downloads/photometry_ZFM-09139_2025-08-05/photometry_signal.pqt' 
# photometry_table = pd.read_parquet(table_path)

# eid = '37215b1d-8baa-48ca-a629-bb38b6c404bc'
#another example 
# eid = '1be2e1e9-e4f2-41ad-9e34-ae27967d41ac'
# region = photometry_table.columns[0]
# Load the behavior
df_trials, subject, session_date = load_trials_updated(eid) 

# print(subject, session_date, region)


import pandas as pd
from pathlib import Path
from iblphotometry import io
# from brainbox.io.one import SessionLoader


class PhotometryLoader:
    # TODO make this class a subclass of SessionLoader
    # TODO move this class to brainbox.io

    def __init__(self, one, verbose=False):
        self.one = one
        self.verbose = verbose

    def load_photometry_data(self, eid=None, pid=None, rename=True) -> pd.DataFrame:
        if pid is not None:
            raise NotImplementedError
            # return self._load_data_from_pid(pid)

        if eid is not None:
            return self._load_data_from_eid(eid, rename=rename)

    def _load_data_from_eid(self, eid, rename=True) -> pd.DataFrame:
        raw_photometry_df = self.one.load_dataset(eid, 'photometry.signal.pqt')
        locations_df = self.one.load_dataset(eid, 'photometryROI.locations.pqt')
        read_config = dict(
            data_columns=list(locations_df.index),
            rename=locations_df['brain_region'].to_dict() if rename else None,
        )
        raw_dfs = io.from_ibl_dataframe(raw_photometry_df, **read_config)

        signal_band_names = list(raw_dfs.keys())
        col_names = list(raw_dfs[signal_band_names[0]].columns)
        if self.verbose:
            print(f'available signal bands: {signal_band_names}')
            print(f'available brain regions: {col_names}')

        return raw_dfs


class KceniaLoader(PhotometryLoader):
    # soon do be OBSOLETE
    def _load_data_from_eid(self, eid: str, rename=True):
        session_path = self.one.eid2path(eid)
        pnames = self._eid2pnames(eid)

        _raw_dfs = {}
        for pname in pnames:
            pqt_path = session_path / 'alf' / pname / 'raw_photometry.pqt'
            _raw_dfs[pname] = pd.read_parquet(pqt_path).set_index('times')

        signal_bands = ['raw_calcium', 'raw_isosbestic']  # HARDCODED but fine

        # flipping the data representation
        raw_dfs = {}
        for band in signal_bands:
            df = pd.DataFrame([_raw_dfs[pname][band].values for pname in pnames]).T
            df.columns = pnames
            df.index = _raw_dfs[pname][band].index
            raw_dfs[band] = df

        if self.verbose:
            print(f'available signal bands: {list(raw_dfs.keys())}')
            cols = list(raw_dfs[list(raw_dfs.keys())[0]].columns)
            print(f'available brain regions: {cols}')

        return raw_dfs

    def _eid2pnames(self, eid: str):
        session_path = self.one.eid2path(eid)
        pnames = [reg.name for reg in session_path.joinpath('alf').glob('Region*')]
        return pnames





# Your eid
# eid = '1be2e1e9-e4f2-41ad-9e34-ae27967d41ac'

# --- Load behavior
df_trials, subject, session_date = load_trials_updated(eid)

print("Behavior table:")
print(df_trials.head())

# --- Load photometry
loader = PhotometryLoader(one, verbose=True)
raw_dfs = loader.load_photometry_data(eid=eid, rename=True)

# raw_dfs is a dict: e.g. {'raw_calcium': df1, 'raw_isosbestic': df2, ...}
print("Photometry bands available:", raw_dfs.keys())

# Example: pick calcium band
df_phot = raw_dfs['GCaMP']
print("Photometry table (GCaMP):")
print(df_phot.head())

# --- Optional: align time ranges
tmin, tmax = df_trials["intervals_0"].min(), df_trials["intervals_1"].max()
df_phot = df_phot.loc[(df_phot.index >= tmin) & (df_phot.index <= tmax)]

print(f"\nExtracted for {subject} on {session_date}:")
print(f"- {len(df_trials)} trials")
print(f"- {df_phot.shape[0]} photometry samples in range")

df_phot = df_phot.reset_index()

df_nph = df_phot
nph = df_nph
nph['zdFF'] = nph.LC

raw_signal = nph['zdFF'][0:]
smooth_win = 10
smooth_signal = smooth_signal(raw_signal, smooth_win) 

lambd = 5e4 # Adjust lambda to get the best fit
porder = 1
itermax = 50
s_base=airPLS(smooth_signal,lambda_=lambd,porder=porder,itermax=itermax)

remove=0
signal = (smooth_signal[remove:] - s_base[remove:])  

z_signal = (signal - np.median(signal)) / np.std(signal)



df_nph['zdFF'] = z_signal

nph = df_nph 

fs = 1 / np.median(np.diff(nph.times.values))
fs



plt.figure(figsize=(20, 8))
plt.plot(nph.times, nph.zdFF, c='teal', alpha=0.8, linewidth=0.15)
for i in df_trials.feedback_times: 
    plt.axvline(x=i, linewidth=0.2, color='black', alpha=0.75) 
plt.show() 


behav = df_trials
photometry_feedback, idx_psth = psth(
    calcium=nph.zdFF.values,
    times=nph.times.values,
    t_events=behav[EVENT].values,
    fs=fs,
    peri_event_window=[-1, 2]
)




PERIEVENT_WINDOW = [-1, 2]

time_axis = np.arange(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], 1/fs)

plt.figure(figsize=(10, 8))
plt.plot(time_axis, photometry_feedback, color='black', linewidth=0.3, alpha=0.3)
plt.axvline(x=0, color='red', linestyle='--')  # Event at 0s
plt.xlabel("Time (s)")
plt.ylabel("ΔF/F (z-scored)")
plt.title("Peri-feedback PSTH")
plt.show()





# %%
"""
MORE PLOTS 
"""
n_timepoints, n_trials = photometry_feedback.shape
time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints, endpoint=False)

# ---------- Mean ± SEM across all trials ----------
mean_trace = np.nanmean(photometry_feedback, axis=1)              # mean across trials
sem_trace  = np.nanstd(photometry_feedback, axis=1) / np.sqrt(n_trials)

plt.figure(figsize=(8, 6))
plt.plot(time_axis, mean_trace, lw=2, label="Mean")
plt.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3, label="SEM")
plt.axvline(0, ls='--')
plt.xlabel("Time (s)")
plt.ylabel("ΔF/F (z-scored)")
plt.title("Peri-event PSTH (mean ± SEM)")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- Trial-by-trial heatmap ----------
# Optional: sort trials by a behavior column (e.g., feedbackType) to make structure clearer
# Remove/disable the next 2 lines if you don't want sorting:
trial_order = np.argsort(behav["feedbackType"].fillna(-999).to_numpy())
psth_for_heat = photometry_feedback[:, trial_order].T  # rows=trials, cols=time

fig, ax = plt.subplots(figsize=(10, 6))
im = sns.heatmap(
    psth_for_heat,
    cmap="rocket",  # or "rocket" if you prefer warmer palette
    center=0,        # zero-centered color map is helpful for z-scored data
    cbar_kws={"label": "ΔF/F (z)"}
)

# Put time ticks in seconds along x-axis
tick_secs = np.arange(np.ceil(PERIEVENT_WINDOW[0]), np.floor(PERIEVENT_WINDOW[1]) + 1, 1.0)
tick_idx  = np.searchsorted(time_axis, tick_secs)
ax.set_xticks(tick_idx)
ax.set_xticklabels([f"{s:.0f}" for s in tick_secs])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Trials")

# Vertical line at event time (0 s)
zero_idx = np.searchsorted(time_axis, 0)
ax.axvline(zero_idx, color="w", lw=1)

ax.set_title("Peri-event photometry (trial heatmap)")
plt.tight_layout()
plt.show()

# %%
""" 
PSTH single trials and mean 
"""
# photometry_feedback: shape [timepoints, n_trials]
n_time, n_trials = photometry_feedback.shape
time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_time, endpoint=False)

# valid trials = columns that aren't all-NaN (out-of-bounds get left as NaN)
valid_cols = ~np.all(np.isnan(photometry_feedback), axis=0)
pf_valid = photometry_feedback[:, valid_cols]
n_valid = pf_valid.shape[1]

fig, ax = plt.subplots(figsize=(10, 8))
# single-trial traces
ax.plot(time_axis, photometry_feedback, linewidth=0.3, alpha=0.3, color='gray')

# mean ± SEM
mean_trace = np.nanmean(pf_valid, axis=1)
sem_trace  = np.nanstd(pf_valid, axis=1) / np.sqrt(n_valid)

ax.plot(time_axis, mean_trace, linewidth=3.5, label="Mean")
ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.25, label="SEM")

ax.axvline(0, linestyle='--')
ax.set_xlabel("Time (s)")
ax.set_ylabel("ΔF/F (z)")
ax.set_title("Peri-event PSTH (single trials + mean ± SEM)")
ax.legend()
plt.tight_layout()
plt.show()

# %%
""" divide correct and incorrect """
ft = behav["feedbackType"].to_numpy()

groups = {
    "Correct (1)":   (ft == 1),
    "Incorrect (-1)": (ft == -1),
}

fig, ax = plt.subplots(figsize=(10, 8))
for label, mask in groups.items():
    if mask.sum() == 0:
        continue
    # select columns (trials) by mask
    traces = photometry_feedback[:, mask]
    # drop columns that are all-NaN
    keep = ~np.all(np.isnan(traces), axis=0)
    traces = traces[:, keep]
    if traces.size == 0:
        continue
    m  = np.nanmean(traces, axis=1)
    se = np.nanstd(traces, axis=1) / np.sqrt(traces.shape[1])
    ax.plot(time_axis, m, linewidth=2, label=f"{label} (n={traces.shape[1]})")
    ax.fill_between(time_axis, m - se, m + se, alpha=0.25)

ax.axvline(0, linestyle='--')
ax.set_xlabel("Time (s)")
ax.set_ylabel("ΔF/F (z)")
ax.set_title("Peri-event PSTH by feedbackType (mean ± SEM)")
ax.legend()
plt.tight_layout()
plt.show()

# %%
ft = behav["feedbackType"].to_numpy()

groups = {
    "Correct (1)":   (ft == 1),
    "Incorrect (-1)": (ft == -1),
}

fig, ax = plt.subplots(figsize=(10, 8))
for label, mask in groups.items():
    if mask.sum() == 0:
        continue
    # select columns (trials) by mask
    traces = photometry_feedback[:, mask]
    # drop columns that are all-NaN
    keep = ~np.all(np.isnan(traces), axis=0)
    traces = traces[:, keep]
    if traces.size == 0:
        continue
    m  = np.nanmean(traces, axis=1)
    se = np.nanstd(traces, axis=1) / np.sqrt(traces.shape[1])
    ax.plot(time_axis, m, linewidth=2, label=f"{label} (n={traces.shape[1]})")
    ax.fill_between(time_axis, m - se, m + se, alpha=0.25)

ax.axvline(0, linestyle='--')
ax.set_xlabel("Time (s)")
ax.set_ylabel("ΔF/F (z)")
ax.set_title("Peri-event PSTH by feedbackType (mean ± SEM)")
ax.legend()
plt.tight_layout()
plt.show()

#%%
# Build a time axis matching the PSTH matrix
n_time, n_trials = photometry_feedback.shape
time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_time, endpoint=False)

# Trial-wise meta
ft     = behav["feedbackType"].to_numpy()      # -1 (incorrect), 1 (correct)
contr  = behav["allContrasts"].to_numpy()

# Lighter grayscale palette (white↔black numbers: 1.0 = white, 0.0 = black)
unique_contrasts = np.sort(np.unique(contr[~np.isnan(contr)]))
n_levels = len(unique_contrasts)

# Make 5 (or n_levels) shades from very light to medium-dark
# e.g., 0.92 (very light gray) → 0.35 (dark gray, not pure black)
shades = np.linspace(0.92, 0.35, n_levels)  # lighter overall, evenly spaced
contrast_to_color = {c: str(shade) for c, shade in zip(unique_contrasts, shades)}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=True)
panels = [("Correct (1)",   ft == 1,  axes[0]),
          ("Incorrect (-1)", ft == -1, axes[1])]

for title, mask_ft, ax in panels:
    # draw lighter first, darker last (so darker lines sit on top)
    for c in unique_contrasts:
        mask_c = np.isclose(contr, c, equal_nan=False)
        col_mask = mask_ft & mask_c
        if not np.any(col_mask):
            continue
        traces = photometry_feedback[:, col_mask]
        keep = ~np.all(np.isnan(traces), axis=0)
        traces = traces[:, keep]
        if traces.size == 0:
            continue

        m  = np.nanmean(traces, axis=1)
        se = np.nanstd(traces, axis=1) / np.sqrt(traces.shape[1])

        color = contrast_to_color[c]        # grayscale string
        ax.plot(time_axis, m, lw=2.5, color=color, label=f"{c:g} (n={traces.shape[1]})")
        ax.fill_between(time_axis, m - se, m + se, alpha=0.18, color=color)

    ax.axvline(0, ls='--')
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (z)")

axes[1].legend(title="allContrasts", loc="best")

# Hard-lock identical y-lims (sharey already syncs, this just guarantees)
yl = (min(axes[0].get_ylim()[0], axes[1].get_ylim()[0]),
      max(axes[0].get_ylim()[1], axes[1].get_ylim()[1]))
for ax in axes:
    ax.set_ylim(yl)

plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

# Build time axis matching PSTH matrix
n_time, n_trials = photometry_feedback.shape
time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_time, endpoint=False)

# Trial-wise meta
ft    = behav["feedbackType"].to_numpy()      # -1 / 1
contr = behav["allContrasts"].to_numpy()

# Contrast levels
unique_contrasts = np.sort(np.unique(contr[~np.isnan(contr)]))

# --- Inferno colormap (higher contrast -> lighter/yellow). Use 'inferno_r' for higher -> darker.
cmap = get_cmap('inferno_r')   # or 'inferno_r'
norm = Normalize(vmin=unique_contrasts.min(), vmax=unique_contrasts.max())

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=True)
panels = [("Correct (1)",   ft == 1,  axes[0]),
          ("Incorrect (-1)", ft == -1, axes[1])]

for title, mask_ft, ax in panels:
    for c in unique_contrasts:
        mask_c   = np.isclose(contr, c, equal_nan=False)
        col_mask = mask_ft & mask_c
        if not np.any(col_mask):
            continue

        traces = photometry_feedback[:, col_mask]
        keep   = ~np.all(np.isnan(traces), axis=0)
        traces = traces[:, keep]
        if traces.size == 0:
            continue

        m  = np.nanmean(traces, axis=1)
        se = np.nanstd(traces, axis=1) / np.sqrt(traces.shape[1])

        color = cmap(norm(c))
        ax.plot(time_axis, m, lw=2.5, color=color, label=f"{c:g} (n={traces.shape[1]})")
        ax.fill_between(time_axis, m - se, m + se, alpha=0.25, color=color)

    ax.axvline(0, ls='--')
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (z)")

# Shared colorbar for contrast levels
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02, shrink=0.9)
cbar.set_label('allContrasts')

# (sharey=True already syncs y; hard-lock just in case)
yl = (min(axes[0].get_ylim()[0], axes[1].get_ylim()[0]),
      max(axes[0].get_ylim()[1], axes[1].get_ylim()[1]))
for ax in axes:
    ax.set_ylim(yl)

plt.show()

# %%
