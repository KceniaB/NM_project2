#%% 
""" 
KB 2025-08-08
VERSION1 - if you have the file: {MOUSE NAME}_{DATE}/_neurophotometrics_fpData_raw.pqt
Code to pick the new [2025] photometry and behavior files and preprocess them
output: preprocessed aligned signal; heatmap; psth 

TO CHANGE: lines 378-381 the file path to those files 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
# from brainbox.behavior.training import compute_performance 
# from brainbox.io.one import SessionLoader 
# import iblphotometry.kcenia as kcenia
import ibldsp.utils
# import scipy.signal
from iblutil.numerical import rcoeff
import sys
# sys.path.insert(0, "/home/kceniabougrova/Documents/GitHub/ibl-photometry/src")

from one.api import ONE #always after the imports
one = ONE() 


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





def get_eid(mouse=mouse,date=date): 
    eids = one.search(subject=mouse, date=date) 
    eid = eids[0]
    ref = one.eid2ref(eid)
    print(eid)
    print(ref) 
    return eid  



#%%
#===========================================================================
#                            Pick data from table 
#===========================================================================
"""
search for those 3 files: 
    - _neurophotometrics_fpData_raw.pqt
    - photometryROI_locations.pqt - or just know the region G from which you're recording from 
    - photometry_signal.pqt
"""
table_path = '/home/kceniabougrova/Downloads/photometry_ZFM-09139_2025-08-05/_neurophotometrics_fpData_raw.pqt' 
photometry_raw_table = pd.read_parquet(table_path)
table_path = '/home/kceniabougrova/Downloads/photometry_ZFM-09139_2025-08-05/photometryROI_locations.pqt' 
region_map = pd.read_parquet(table_path)
# table_path = '/home/kceniabougrova/Downloads/photometry_ZFM-09139_2025-08-05/photometry_signal.pqt' 
# photometry_table = pd.read_parquet(table_path) #THIS IS VERSION2 

roi_name = region_map.index[0]  # picks 'G2' in your example
cols_to_keep = ["SystemTimestamp", "LedState", "ComputerTimestamp", roi_name]

new_photometry_raw_table = photometry_raw_table[cols_to_keep].copy() 

eid = 'da728f9d-a116-4746-bd5a-c25006d78fca'

# another example
table_path = '/home/kceniabougrova/Downloads/photometryROI.locations.50b4475e-230b-46c3-8959-91e1d4b5e34e.pqt' 
region_map = pd.read_parquet(table_path)
table_path = '/home/kceniabougrova/Downloads/photometry.signal.3070465c-c0df-4585-bd7a-bf50465d136e.pqt'
photometry_table = pd.read_parquet(table_path) #THIS IS VERSION2 
mouse = 'ZFM-08751'
date = '2025-06-12'
eid = get_eid(mouse=mouse,date=date) #73636c9d-1bf6-40db-a661-4a648e3b100b




roi_name = region_map.index[0]  # picks 'G2' in your example

region = roi_name
# Load the behavior
df_trials, subject, session_date = load_trials_updated(eid) 

print(subject, session_date, region)



#%% #########################################################################################################
# """ GET PHOTOMETRY DATA """ 


photometry_raw_table["mouse"] = subject
photometry_raw_table["date"] = session_date
photometry_raw_table["region"] = region
photometry_raw_table["eid"] = eid 

# Remove 'Value.' prefix from columns
df_bnc.columns = [col.replace("Value.", "") for col in df_bnc.columns]
df_bnc = df_bnc[df_bnc["Value"] == True].reset_index(drop=True)

tph = df_bnc["Timestamp"].values 





#%%

"""
CHANGE INPUT AUTOMATICALLY 
""" 
# tph = (df_nph['Timestamp'].values[iup] + df_nph['Timestamp'].values[iup - 1]) / 2 #nph TTL times computed for the midvalue 
# tbpod = np.sort(np.r_[
#     df_trials['intervals_0'].values,
#     df_trials['intervals_1'].values - 1,  # here is the trick
#     df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
# )
# fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)
# df_nph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_nph["Timestamp"]) 
# fcn_nph_to_bpod_times(df_nph["Timestamp"])


tbpod = df_trials['stimOnTrigger_times'].values #bpod TTL times

best_cols, match_scores = find_matching_trials_column(df_trials, tph)


"""
CHANGE INPUT AUTOMATICALLY 
"""
# iup = ibldsp.utils.rises(df_nph[f'Input{df_bnc}'].values) #idx nph TTL times 
# tph = (df_nph['Timestamp'].values[iup] + df_nph['Timestamp'].values[iup - 1]) / 2 #nph TTL times computed for the midvalue 
#KB commented two previous and added the next line 

fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True) #interpolation 
if len(tph)/len(tbpod) < .9: 
    print("mismatch in sync, will try to add ITI duration to the sync")
    tbpod = np.sort(np.r_[
        df_trials['intervals_0'].values,
        df_trials['intervals_1'].values - 1,  # here is the trick
        df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
    )
    fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)
    if len(tph)/len(tbpod) > .9:
        print("still mismatch, maybe this is an old session")
        tbpod = np.sort(np.r_[df_trials['stimOnTrigger_times'].values])
        fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True, return_indices=True) 
        assert len(iph)/len(tbpod) > .9
        print("recovered from sync mismatch, continuing #2")
assert abs(drift_ppm) < 100, "drift is more than 100 ppm"

df_nph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_nph["Timestamp"]) 

fcn_nph_to_bpod_times(df_nph["Timestamp"])

df_nph["Timestamp"]




# df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 
session_start = df_trials.intervals_0.values[0] - 10  # Start time, 100 seconds before the first tph value
session_end = df_trials.intervals_1.values[-1] + 10   # End time, 100 seconds after the last tph value

# Select data within the specified time range
selected_data = df_nph[
    (df_nph['bpod_frame_times'] >= session_start) &
    (df_nph['bpod_frame_times'] <= session_end)
] 
df_nph = selected_data.reset_index(drop=True) 

#%%
#===========================================================================
#      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
#===========================================================================
df_nph = LedState_or_Flags(df_nph)

""" 4.1.2 Check for LedState/previous Flags bugs """ 
""" 4.1.2.1 Length """
# Verify the length of the data of the 2 different LEDs
df_470, df_415 = verify_length(df_nph)
""" 4.1.2.2 Verify if there are repeated flags """ 
verify_repetitions(df_nph["LedState"])
""" 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
# session_day=rec.date
# plot_outliers(df_470,df_415,region,mouse,session_day) 

df_ph_1 = df_nph

# Remove rows with LedState 1 at both ends if present
if df_ph_1['LedState'].iloc[0] == 1 and df_ph_1['LedState'].iloc[-1] == 1:
    df_ph_1 = df_ph_1.iloc[1:]

# Remove rows with LedState 2 at both ends if present
if df_ph_1['LedState'].iloc[0] == 2 and df_ph_1['LedState'].iloc[-1] == 2:
    df_ph_1 = df_ph_1.iloc[:-2]

# Filter data for LedState 2 (470nm)
df_470 = df_ph_1[df_ph_1['LedState'] == 2]

# Filter data for LedState 1 (415nm)
df_415 = df_ph_1[df_ph_1['LedState'] == 1]

# Check if the lengths of df_470 and df_415 are equal
assert len(df_470) == len(df_415), "Sync arrays are of different lengths"

# Plot the data
plt.rcParams["figure.figsize"] = (8, 5)
plt.plot(df_470[region], c='#279F95', linewidth=0.5)
plt.plot(df_415[region], c='#803896', linewidth=0.5)
plt.title("Cropped signal "+subject+' '+str(session_date))
plt.legend(["GCaMP", "isosbestic"], frameon=False)
sns.despine(left=False, bottom=False)
plt.show(block=False)
plt.close() 


# Print counts
print("470 =", df_470['LedState'].count(), " 415 =", df_415['LedState'].count())

df_nph = df_ph_1.reset_index(drop=True)  
df_470 = df_nph[df_nph.LedState==2] 
df_470 = df_470.reset_index(drop=True)
df_415 = df_nph[df_nph.LedState==1] 
df_415 = df_415.reset_index(drop=True) 
#================================================
""" 4.1.4 FRAME RATE """ 
time_diffs = (df_470["Timestamp"]).diff().dropna() 
fs = 1 / time_diffs.median() 

raw_reference = df_415[region] #isosbestic 
raw_signal = df_470[region] #GCaMP signal 
raw_timestamps_bpod = df_470["bpod_frame_times"]
raw_timestamps_nph_470 = df_470["Timestamp"]
raw_timestamps_nph_415 = df_415["Timestamp"]
raw_TTL_bpod = tbpod
raw_TTL_nph = tph

# my_array = np.c_[raw_timestamps_bpod, raw_reference, raw_signal]
my_array = np.column_stack((raw_timestamps_bpod, raw_reference, raw_signal))

df_nph = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium']) #IMPORTANT DF


plt.figure(figsize=(20, 6))

# Plot calcium and isosbestic signals
plt.plot(df_nph['times'][200:1000], df_nph['raw_calcium'][200:1000], linewidth=1.25, alpha=0.8, color='teal') 
plt.plot(df_nph['times'][200:1000], df_nph['raw_isosbestic'][200:1000], linewidth=1.25, alpha=0.8, color='purple') 

# Vertical lines at stimulus onset
for t in df_trials['stimOnTrigger_times'].dropna():
    if df_nph['times'].iloc[200] <= t <= df_nph['times'].iloc[999]:
        plt.axvline(t, color='gray', linestyle='--', alpha=0.4, linewidth=1)

# Vertical lines at feedback times, color-coded by feedbackType
for t, fb_type in zip(df_trials['feedback_times'], df_trials['feedbackType']):
    if pd.notna(t) and df_nph['times'].iloc[200] <= t <= df_nph['times'].iloc[999]:
        color = 'blue' if fb_type == 1 else 'red' if fb_type == -1 else 'gray'
        plt.axvline(t, color=color, linestyle='-', alpha=0.6, linewidth=1.5)

plt.tight_layout()
plt.show()










#%%


raw_reference = df_nph['raw_isosbestic'][0:]
raw_signal = df_nph['raw_calcium'][0:]

smooth_win = 10
smooth_reference = smooth_signal(raw_reference, smooth_win)
smooth_signal = smooth_signal(raw_signal, smooth_win) 

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(smooth_signal,'blue',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(smooth_reference,'purple',linewidth=1.5)


#%%

lambd = 5e4 # Adjust lambda to get the best fit
porder = 1
itermax = 50
r_base=airPLS(smooth_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
s_base=airPLS(smooth_signal,lambda_=lambd,porder=porder,itermax=itermax)

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(smooth_signal,'blue',linewidth=1.5)
ax1.plot(s_base,'black',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(smooth_reference,'purple',linewidth=1.5)
ax2.plot(r_base,'black',linewidth=1.5)



#%%

remove=0
reference = (smooth_reference[remove:] - r_base[remove:])
signal = (smooth_signal[remove:] - s_base[remove:])  

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(signal,'blue',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(reference,'purple',linewidth=1.5)





#%%

z_reference = (reference - np.median(reference)) / np.std(reference)
z_signal = (signal - np.median(signal)) / np.std(signal)

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(z_signal,'blue',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(z_reference,'purple',linewidth=1.5)



#%%
from sklearn.linear_model import Lasso
lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
            positive=True, random_state=9999, selection='random')
n = len(z_reference)
lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))

z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(z_reference,z_signal,'b.')
ax1.plot(z_reference,z_reference_fitted, 'r--',linewidth=1.5)



#%%
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(z_signal,'blue')
ax1.plot(z_reference_fitted,'purple')




#%%
zdFF = (z_signal - z_reference_fitted)


#%%
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(zdFF,'black')

#%%
df_nph['zdFF'] = zdFF
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
    t_events=behav["intervals_0"].values,
    fs=fs,
    peri_event_window=[-1, 2]
)



plt.figure(figsize=(15, 8))
plt.plot(photometry_feedback, color='black', linewidth=0.3, alpha=0.3)
plt.axvline(x=photometry_feedback.shape[0] // 3, color='red', linestyle='--')  # Event at t=0
plt.xlabel("Timepoints")
plt.ylabel("ΔF/F (z-scored)")
plt.title("Peri-feedback PSTH")
plt.show()


PERIEVENT_WINDOW = [-1, 2]

time_axis = np.arange(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], 1/fs)

plt.figure(figsize=(15, 8))
plt.plot(time_axis, photometry_feedback, color='black', linewidth=0.3, alpha=0.3)
plt.axvline(x=0, color='red', linestyle='--')  # Event at 0s
plt.xlabel("Time (s)")




# %% 


""" SELECT THE EVENT AND WHAT INTERVAL TO PLOT IN THE PSTH """ 
EVENT = "feedback_times" 
time_bef = -1
time_aft = 2
PERIEVENT_WINDOW = [time_bef,time_aft]
SAMPLING_RATE = int(1/np.mean(np.diff(df_nph.times))) 

array_timestamps = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps, event_test) #check idx where they would be included, in a sorted way 
""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
df_nph.loc[idx_event,"trial_number"]=1
df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) 

event_times = np.array(df_trials[EVENT]) #pick the feedback timestamps 

event_idx = np.searchsorted(array_timestamps, event_times) #check idx where they would be included, in a sorted way 

psth_idx += event_idx


#%%

def plot_heatmap_psth(preprocessingtype=df_nph.zdFF): 
    psth_good = preprocessingtype.values[psth_idx[:,(df_trials.feedbackType == 1)]]
    psth_error = preprocessingtype.values[psth_idx[:,(df_trials.feedbackType == -1)]]
    # Calculate averages and SEM
    psth_good_avg = psth_good.mean(axis=1)
    sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
    psth_error_avg = psth_error.mean(axis=1)
    sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

    # Create the figure and gridspec
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    # Plot the heatmap and line plot for correct trials
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
    ax1.invert_yaxis()
    ax1.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax1.set_title('Correct Trials')

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3) 
    # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
    ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
    ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax2.set_ylabel('Average Value')
    ax2.set_xlabel('Time')

    # Plot the heatmap and line plot for incorrect trials
    ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
    sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
    ax3.invert_yaxis()
    ax3.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax3.set_title('Incorrect Trials')

    ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)
    ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
    ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
    ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax4.set_ylabel('Average Value')
    ax4.set_xlabel('Time')

    fig.suptitle(f'zdFF_{EVENT}_{subject}_{session_date}_{region}_{eid}', y=1, fontsize=14)
    plt.tight_layout()
    # plt.savefig(f'/mnt/h0/kb/data/psth_npy/Fig02_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
    plt.show() 

plot_heatmap_psth(df_nph.zdFF)



# %%
""" 
to sort by: reactionTimes 
""" 

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap_psth_sorted_by_rt(preprocessingtype=df_nph.zdFF):
    # Boolean masks
    mask_correct = df_trials.feedbackType == 1
    mask_incorrect = df_trials.feedbackType == -1

    # PSTHs (n_timepoints × n_trials)
    psth_good = preprocessingtype.values[psth_idx[:, mask_correct]]
    psth_error = preprocessingtype.values[psth_idx[:, mask_incorrect]]

    # Reaction times and sort indices
    reaction_times_correct = df_trials.loc[mask_correct, 'reactionTime'].values
    sort_idx_correct = np.argsort(reaction_times_correct)
    psth_good_sorted = psth_good.T[sort_idx_correct]  # transpose → trials × time
    reaction_times_correct_sorted = reaction_times_correct[sort_idx_correct]

    reaction_times_incorrect = df_trials.loc[mask_incorrect, 'reactionTime'].values
    sort_idx_incorrect = np.argsort(reaction_times_incorrect)
    psth_error_sorted = psth_error.T[sort_idx_incorrect]
    reaction_times_incorrect_sorted = reaction_times_incorrect[sort_idx_incorrect]

    # Round for tick display
    rt_correct_rounded = np.round(reaction_times_correct_sorted, 2)
    rt_incorrect_rounded = np.round(reaction_times_incorrect_sorted, 2)

    # Average and SEM
    psth_good_avg = psth_good.mean(axis=1)
    sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
    psth_error_avg = psth_error.mean(axis=1)
    sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

    # Plot
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    # === Correct trials heatmap ===
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(psth_good_sorted, cbar=False, ax=ax1)
    ax1.invert_yaxis()
    ax1.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax1.set_title('Correct Trials (sorted by RT)')

    # Set y-ticks with RT labels
    step = 5  # label every 5th trial
    tick_locs = np.arange(0, len(rt_correct_rounded), step)
    ax1.set_yticks(tick_locs + 0.5)
    ax1.set_yticklabels(rt_correct_rounded[tick_locs])
    ax1.set_ylabel("Reaction time (s)")

    # === Correct average plot ===
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3)
    ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
    ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax2.set_ylabel('Average Value')
    ax2.set_xlabel('Time')

    # === Incorrect trials heatmap ===
    ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
    sns.heatmap(psth_error_sorted, cbar=False, ax=ax3)
    ax3.invert_yaxis()
    ax3.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax3.set_title('Incorrect Trials (sorted by RT)')

    tick_locs_err = np.arange(0, len(rt_incorrect_rounded), step)
    ax3.set_yticks(tick_locs_err + 0.5)
    ax3.set_yticklabels(rt_incorrect_rounded[tick_locs_err])
    ax3.set_ylabel("Reaction time (s)")

    # === Incorrect average plot ===
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)
    ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
    ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
    ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax4.set_ylabel('Average Value')
    ax4.set_xlabel('Time')

    fig.suptitle(f'zdFF_sortedByRT_{EVENT}_{subject}_{session_date}_{region}_{eid}', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()



#%% 

"""
Plot the same, but all the code to choose the event and plot the heatmap 
to change: 
EVENT
responseTime
""" 
""" SELECT THE EVENT AND WHAT INTERVAL TO PLOT IN THE PSTH """ 
EVENT = "stimOnTrigger_times" 
time_bef = -2
time_aft = 2
PERIEVENT_WINDOW = [time_bef,time_aft]
SAMPLING_RATE = int(1/np.mean(np.diff(df_nph.times))) 

array_timestamps = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps, event_test) #check idx where they would be included, in a sorted way 
""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
df_nph.loc[idx_event,"trial_number"]=1
df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) 

event_times = np.array(df_trials[EVENT]) #pick the feedback timestamps 

event_idx = np.searchsorted(array_timestamps, event_times) #check idx where they would be included, in a sorted way 

psth_idx += event_idx


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap_psth_sorted_by_rt(preprocessingtype=df_nph.zdFF, column_sort='reactionTime'):
    # Boolean masks
    mask_correct = df_trials.feedbackType == 1
    mask_incorrect = df_trials.feedbackType == -1

    # PSTHs (n_timepoints × n_trials)
    psth_good = preprocessingtype.values[psth_idx[:, mask_correct]]
    psth_error = preprocessingtype.values[psth_idx[:, mask_incorrect]]

    # Reaction times and sort indices
    reaction_times_correct = df_trials.loc[mask_correct, column_sort].values
    sort_idx_correct = np.argsort(reaction_times_correct)
    psth_good_sorted = psth_good.T[sort_idx_correct]  # transpose → trials × time
    reaction_times_correct_sorted = reaction_times_correct[sort_idx_correct]

    reaction_times_incorrect = df_trials.loc[mask_incorrect, column_sort].values
    sort_idx_incorrect = np.argsort(reaction_times_incorrect)
    psth_error_sorted = psth_error.T[sort_idx_incorrect]
    reaction_times_incorrect_sorted = reaction_times_incorrect[sort_idx_incorrect]

    # Round for tick display
    rt_correct_rounded = np.round(reaction_times_correct_sorted, 2)
    rt_incorrect_rounded = np.round(reaction_times_incorrect_sorted, 2)

    # Average and SEM
    psth_good_avg = psth_good.mean(axis=1)
    sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
    psth_error_avg = psth_error.mean(axis=1)
    sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

    # Plot
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    # === Correct trials heatmap ===
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(psth_good_sorted, cbar=False, ax=ax1)
    ax1.invert_yaxis()
    ax1.axvline(x=60, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax1.set_title(f'Correct Trials (sorted by {column_sort})')

    # Set y-ticks with RT labels
    step = 5  # label every 5th trial
    tick_locs = np.arange(0, len(rt_correct_rounded), step)
    ax1.set_yticks(tick_locs + 0.5)
    ax1.set_yticklabels(rt_correct_rounded[tick_locs])
    ax1.set_ylabel(f"{column_sort} (s)")

    # === Correct average plot ===
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3)
    ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
    ax2.axvline(x=60, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax2.set_ylabel('Average Value')
    ax2.set_xlabel('Time')

    # === Incorrect trials heatmap ===
    ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
    sns.heatmap(psth_error_sorted, cbar=False, ax=ax3)
    ax3.invert_yaxis()
    ax3.axvline(x=60, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax3.set_title(f'Incorrect Trials (sorted by {column_sort})')

    tick_locs_err = np.arange(0, len(rt_incorrect_rounded), step)
    ax3.set_yticks(tick_locs_err + 0.5)
    ax3.set_yticklabels(rt_incorrect_rounded[tick_locs_err])
    ax3.set_ylabel(f"{column_sort} (s)")

    # === Incorrect average plot ===
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)
    ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
    ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
    ax4.axvline(x=60, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax4.set_ylabel('Average Value')
    ax4.set_xlabel('Time')

    fig.suptitle(f'zdFF_sortedBy{column_sort}_{EVENT}_{subject}_{session_date}_{region}_{eid}', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

plot_heatmap_psth_sorted_by_rt(df_nph.zdFF, column_sort='quiescenceTime')

# %%
"""
prediction for the vars that wuld better explain the drop in the 5-HT activity before the stimOnTrigger_times 
"""
# Define window before stim (e.g., -0.5s to 0s)
pre_window = [-0.5, 0]  # in seconds

# Create empty list for signal drops
zdff_drops = []

for _, trial in df_trials.iterrows():
   stim_time = trial['stimOnTrigger_times']
  
   # Define time window for this trial
   start = stim_time + pre_window[0]
   end = stim_time + pre_window[1]

   # Mask for that time window
   mask = (df_nph['times'] >= start) & (df_nph['times'] < end)

   # Extract zdFF segment
   segment = df_nph.loc[mask, 'zdFF']

   # Measure "drop" — e.g., min value or average
   drop = segment.mean() if not segment.empty else np.nan
   zdff_drops.append(drop)

# Add to df_trials
df_trials['zdff_drop_pre_stim'] = zdff_drops

df_trials['correct'] = df_trials['feedbackType'] == 1
df_trials['consecutive_correct'] = df_trials['correct'].astype(int).rolling(window=2).sum().fillna(0)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Choose predictors
X = df_trials[['quiescenceTime', 'trialTime', 'reactionTime', 'consecutive_correct']]
X = X.fillna(0)  # handle NaNs

# Add correctness as binary
X['correct'] = (df_trials['feedbackType'] == 1).astype(int)

# Target
y = df_trials['zdff_drop_pre_stim']

# Drop NaNs
mask = y.notna()
X = X[mask]
y = y[mask]

# Scale X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit model
model = LinearRegression()
model.fit(X_scaled, y)

# Print coefficients
coeffs = pd.Series(model.coef_, index=X.columns)
print("Model coefficients:\n", coeffs.sort_values(key=abs, ascending=False))

import seaborn as sns
sns.scatterplot(data=df_trials, x='quiescenceTime', y='zdff_drop_pre_stim', hue='feedbackType')

sns.pairplot(df_trials, vars=['zdff_drop_pre_stim', 'quiescenceTime', 'trialTime', 'reactionTime', 'consecutive_correct'], hue='correct')





#%%
# First, a binary vector of correct trials
correct = (df_trials['feedbackType'] == 1).astype(int)
incorrect = (df_trials['feedbackType'] == -1).astype(int)

# Now, compute cumulative streaks and shift them
prev_consecutive_correct = [] 
prev_consecutive_incorrect = [] 

streak = 0
for i in range(len(correct)):
    prev_consecutive_correct.append(streak)
    if correct[i] == 1:
        streak += 1
    else:
        streak = 0

streak = 0
for i in range(len(incorrect)):
    prev_consecutive_incorrect.append(streak)
    if incorrect[i] == 1:
        streak += 1
    else:
        streak = 0

df_trials['prev_consecutive_correct'] = prev_consecutive_correct
df_trials['prev_consecutive_incorrect'] = prev_consecutive_incorrect

df_trials['prev_feedbackType'] = df_trials['feedbackType'].shift(1)
df_trials['prev_correct'] = (df_trials['feedbackType'].shift(1) == 1).astype(int)
df_trials['prev_trialTime'] = df_trials['trialTime'].shift(1)
df_trials['prev_quiescenceTime'] = df_trials['quiescenceTime'].shift(1)
df_trials['prev_choice'] = df_trials['choice'].shift(1)
df_trials['prev_contrast'] = df_trials["allContrasts"]
df_trials['prev_reactionTime'] = df_trials['reactionTime'].shift(1)

# e.g., number of corrects in last 3 trials
df_trials['rolling_correct_3'] = df_trials['feedbackType'].rolling(3).apply(lambda x: np.sum(x == 1)).shift(1)

features = [
    'quiescenceTime', 'trialTime', 'reactionTime', 'choice', 'correct',
    'consecutive_correct', 'consecutive_incorrect',
    'prev_correct', 'prev_feedbackType', 'prev_trialTime',
    'prev_quiescenceTime', 'prev_reactionTime', 'prev_choice', 'prev_contrast', 'prev_reactionTime', 'rolling_correct_3', "prev_consecutive_correct", "prev_consecutive_incorrect"
]

X = df_trials[features]
X = X.fillna(0)  # fill NaNs (first trial, etc.)
y = df_trials['zdff_drop_pre_stim']

# Drop NaNs in y
mask = y.notna()
X = X[mask]
y = y[mask]

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

import pandas as pd
coeffs = pd.Series(model.coef_, index=X.columns)
print(coeffs.sort_values(key=abs, ascending=False))


import matplotlib.pyplot as plt

# Sort by absolute value
sorted_coeffs = coeffs.sort_values(key=abs, ascending=True)

plt.figure(figsize=(10, 8))
sorted_coeffs.plot(kind='barh', color=(sorted_coeffs > 0).map({True: 'seagreen', False: 'indianred'}))
plt.xlabel("Standardized Coefficient")
plt.title("Feature Importance in Predicting zdFF Drop Pre-Stim")
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()




#%% 


X['correct'] = X['correct'].astype(int)

print(X.dtypes)
print(X.columns[X.dtypes == 'object'])  # This should be empty

from sklearn.model_selection import train_test_split
import xgboost as xgb

# Remove NaNs in target
y = df_trials['zdff_drop_pre_stim']
mask = y.notna()
X = X[mask]
y = y[mask]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

"""
feature importance 
"""
import matplotlib.pyplot as plt
import pandas as pd

# Feature importance
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 10))
feat_imp.plot(kind='barh')
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()







# %%
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Create previous-trial features
df_trials['prev_probabilityLeft'] = df_trials['probabilityLeft'].shift(1)
df_trials['prev_feedbackType'] = df_trials['feedbackType'].shift(1)
df_trials['prev_trialTime'] = df_trials['trialTime'].shift(1)
df_trials['prev_responseTime'] = df_trials['responseTime'].shift(1)
df_trials['prev_choice'] = df_trials['choice'].shift(1)

# Create new feature: stimOn - quiescence start
df_trials['new_stim_minus_quiesc'] = df_trials['stimOnTrigger_times'] - df_trials['quiescenceTime']

# Define variables
selected_vars = [
    'probabilityLeft', 'feedbackType', 'trialTime', 'responseTime',
    'choice', 'quiescenceTime', 'zdff_drop_pre_stim', 'correct', 'consecutive_correct',
    'prev_probabilityLeft', 'prev_feedbackType', 'prev_trialTime',
    'prev_responseTime', 'prev_choice', 'new_stim_minus_quiesc'
]

# Clean up DataFrame
df_clean = df_trials[selected_vars].copy()
df_clean = df_clean.dropna()

# Define predictors and target
X = df_clean.drop(columns='zdff_drop_pre_stim')
y = df_clean['zdff_drop_pre_stim']

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit linear model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Coefficients
coeffs = pd.Series(model.coef_, index=X.columns)
coeffs_sorted = coeffs.sort_values(key=abs, ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=coeffs_sorted.values, y=coeffs_sorted.index, palette='coolwarm')
plt.title(f'Predictors of zdFF drop before stimOn\nR² = {r2:.3f} | MSE = {mse:.3f}')
plt.xlabel('Linear Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()





#%%
