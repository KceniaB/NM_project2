""" Functions """
"""
2025-07-07

Functions to use within this repo

"""
import pandas as pd
import numpy as np

from one.api import ONE #always after the imports 
# one = ONE(cache_dir="/mnt/h0/kb/data/one") 
one = ONE() 


""" useful""" 
# eids = one.search(project='ibl_fibrephotometry') 


""" LOAD TRIALS """ 

def select_one_session(sessions_list, row=150):
    eid = sessions_list.loc[row, "eid"]
    subject = sessions_list.loc[row, "subject"]
    date = sessions_list.loc[row, "date"]
    region = sessions_list.loc[row, "region"]
    nph_file_path = sessions_list.loc[row, "photometry_path_a"]
    nph_bnc_path = sessions_list.loc[row, "digital_inputs_path"]
    return eid, subject, date, region, nph_file_path, nph_bnc_path

def find_matching_trials_column(df_trials, tph, verbose=True, max_ratio_diff=0.1):
    """
    Finds one or more df_trials columns whose time series (ending in '_times') 
    best match the timestamps in `tph` in terms of length and time difference patterns.
    
    Returns:
        - best_cols: list of matching column names
        - match_scores: dict of match scores for all _times columns
    """
    tph = np.asarray(tph)
    tph_diffs = np.diff(tph)
    
    match_scores = {}
    len_ratios = {}
    cols = [col for col in df_trials.columns if col.endswith("_times")]

    for col in cols:
        col_vals = df_trials[col].dropna().values
        len_ratio = len(tph) / len(col_vals)
        len_ratios[col] = len_ratio

        if abs(len_ratio - 1) > max_ratio_diff and abs(len_ratio - 1) > (1 / len(col_vals)): 
            continue  # length mismatch too large

        # Compare time interval patterns
        d_col = np.diff(col_vals)
        min_len = min(len(tph_diffs), len(d_col))
        score = np.corrcoef(tph_diffs[:min_len], d_col[:min_len])[0, 1]
        match_scores[col] = score

    # Sort by score
    sorted_cols = sorted(match_scores.items(), key=lambda x: -abs(x[1]))
    best_cols = [col for col, score in sorted_cols if abs(score) > 0.9]

    if verbose:
        print("==== Matching _times columns ====")
        for col, score in sorted_cols:
            print(f"{col}: corr = {score:.3f}, len ratio = {len_ratios[col]:.2f}")
        if not best_cols:
            print("⚠️ No clear match found.")
        elif len(best_cols) == 1:
            print(f"✅ Best match: {best_cols[0]}")
        else:
            print(f"✅ Multiple likely matches: {best_cols}")

    return best_cols, match_scores

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









def get_regions(rec): 
    """ 
    extracts in string format the mouse name, date of the session, nph file number, bnc file number and regions
    """
    regions = [f"Region{rec.region}G"] 
    return regions


def get_nph(source_path, rec): 
    # source_folder = (f"/home/kceniabougrova/Documents/nph/{rec.date}/")
    source_folder = source_path
    df_nph = pd.read_csv(source_folder+f"raw_photometry{rec.nph_file}.csv") 
    df_nphttl = pd.read_csv(source_folder+f"bonsai_DI{rec.nph_bnc}{rec.nph_file}.csv") 
    return df_nph, df_nphttl 

def get_eid(rec): 
    eids = one.search(subject=rec.mouse, date=rec.date) 
    eid = eids[0]
    ref = one.eid2ref(eid)
    print(eid)
    print(ref) 
    # session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{rec.mouse}/{rec.date}/001/' 
    base_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{rec.mouse}/{rec.date}/' 
    session_path_pattern = f'{base_path}00*/'
    session_paths = glob.glob(session_path_pattern)
    if session_paths:
        session_path_behav = session_paths[0]  # or handle multiple matches as needed
    else:
        session_path_behav = None  # or handle the case where no matching path is found
    file_path = '/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-04022/2022-12-30/001/alf/_ibl_trials.table.pqt'
    df = pd.read_parquet(file_path)




    
    df_alldata = extract_all(session_path_behav)
    table_data = df_alldata[0]['table']
    trials = pd.DataFrame(table_data) 
    return eid, trials 
    
def get_ttl(df_DI0, df_trials): 
    if 'Value.Value' in df_DI0.columns: #for the new ones
        df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
    else:
        df_DI0["Timestamp"] = df_DI0["Seconds"] #for the old ones
    #use Timestamp from this part on, for any of the files
    raw_phdata_DI0_true = df_DI0[df_DI0.Value==True]
    df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
    # raw_phdata_DI0_true = pd.DataFrame(df_DI0.Timestamp[df_DI0.Value==True], columns=['Timestamp'])
    df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True) 
    tph = df_raw_phdata_DI0_T_timestamp.values[:, 0] 
    tbpod = np.sort(np.r_[df_trials['intervals_0'].values, df_trials['intervals_1'].values, df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values])
    return tph, tbpod 



def start_2_end_1(df_photometry): 
    """
    input = raw photometry data
    output = photometry dataframe without the initial flag=0, starting at flag=2, finishing at flag=1, reset_index applied 
    """
    df_photometry = df_photometry.reset_index(drop=True)
    array1 = df_photometry
    if array1["LedState"][0] == 0: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if (array1["LedState"][0] != 2) or (array1["LedState"][0] != 1): 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["LedState"][0] == 1: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["LedState"][len(array1)-1] == 2: 
        array1 = array1[0:len(array1)-1] 
        array1 = array1.reset_index(drop=True)
    array2 = pd.DataFrame(array1)
    return(array2) 
def start_17_end_18(df_photometry): 
    """
    input = raw photometry data
    output = photometry dataframe without the initial flag=16, starting at flag=17, finishing at flag=18, reset_index applied 
    """
    df_photometry = df_photometry.reset_index(drop=True)
    array1 = df_photometry
    if array1["Flags"][0] == 16: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["Flags"][0] == 18: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["Flags"][len(array1)-1] == 17: 
        array1 = array1[0:len(array1)-1] 
        array1 = array1.reset_index(drop=True)
    array2 = pd.DataFrame(array1)
    return(array2) 
""" 4.1.1 Change the Flags that are combined to Flags that will represent only the LED that was on """ 
"""1 and 17 are isosbestic; 2 and 18 are GCaMP"""
def change_flags(df_with_flags): 
    df_with_flags = df_with_flags.reset_index(drop=True)
    if 'LedState' in df_with_flags.columns: 
        array1 = np.array(df_with_flags["LedState"])
        for i in range(0,len(df_with_flags)): 
            if array1[i] == 529 or array1[i] == 273 or array1[i] == 785 or array1[i] == 17: 
                array1[i] = 1
            elif array1[i] == 530 or array1[i] == 274 or array1[i] == 786 or array1[i] == 18: 
                array1[i] = 2
            else: 
                array1[i] = array1[i] 
        array2 = pd.DataFrame(array1)
        df_with_flags["LedState"] = array2
        return(df_with_flags) 
    else: 
        array1 = np.array(df_with_flags["Flags"])
        for i in range(0,len(df_with_flags)): 
            if array1[i] == 529 or array1[i] == 273 or array1[i] == 785 or array1[i] == 17: 
                array1[i] = 1
            elif array1[i] == 530 or array1[i] == 274 or array1[i] == 786 or array1[i] == 18: 
                array1[i] = 2
            else: 
                array1[i] = array1[i] 
        array2 = pd.DataFrame(array1)
        df_with_flags["Flags"] = array2
        return(df_with_flags) 
















#%%

def LedState_or_Flags(df_PhotometryData): 
    if 'LedState' in df_PhotometryData.columns:                         #newversion 
        df_PhotometryData = start_2_end_1(df_PhotometryData)
        df_PhotometryData = df_PhotometryData.reset_index(drop=True)
        df_PhotometryData = (change_flags(df_PhotometryData))
    else:                                                               #oldversion
        df_PhotometryData = start_17_end_18(df_PhotometryData) 
        df_PhotometryData = df_PhotometryData.reset_index(drop=True) 
        df_PhotometryData = (change_flags(df_PhotometryData))
        df_PhotometryData["LedState"] = df_PhotometryData["Flags"]
    return df_PhotometryData


def verify_length(df_PhotometryData): 
    """
    Checking if the length is different
    x = df_470
    y = df_415
    """ 
    x = df_PhotometryData[df_PhotometryData.LedState==2]
    y = df_PhotometryData[df_PhotometryData.LedState==1] 
    if len(x) == len(y): 
        print("Option 1: same length :)")
    else: 
        print("Option 2: SOMETHING IS WRONG! Different len's") 
    print("470 = ",x.LedState.count()," 415 = ",y.LedState.count())
    return(x,y)


def verify_repetitions(x): 
    """
    Checking if there are repetitions in consecutive rows
    x = df_PhotometryData["Flags"]
    """ 
    for i in range(1,(len(x)-1)): 
        if x[i-1] == x[i]: 
            print("here: ", i)



def find_FR(x): 
    """
    find the frame rate of acquisition
    x = df_470["Timestamp"]
    """
    acq_FR = round(1/np.mean(x.diff()))
    # check to make sure that it is 15/30/60! (- with a loop)
    if acq_FR == 30 or acq_FR == 60 or acq_FR == 120: 
        print("All good, the FR is: ", acq_FR)
    else: 
        print("CHECK FR!!!!!!!!!!!!!!!!!!!!") 
    return acq_FR 








# %% 
""" 
Different pre-processing methods for the photometry signal
"""
def jove2019(raw_calcium, raw_isosbestic, fs, **params):
    """
    Martianova, Ekaterina, Sage Aronson, and Christophe D. Proulx. "Multi-fiber photometry to record neural activity in freely-moving animals." JoVE (Journal of Visualized Experiments) 152 (2019): e60278.
    :param raw_calcium:
    :param raw_isosbestic:
    :param params:
    :return:
    """
    # the first step is to remove the photobleaching w
    sos = scipy.signal.butter(fs=fs, output='sos', **params.get('butterworth_lowpass', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}))
    calcium = raw_calcium - scipy.signal.sosfiltfilt(sos, raw_calcium)
    isosbestic = raw_isosbestic - scipy.signal.sosfiltfilt(sos, raw_isosbestic)
    calcium = (calcium - np.median(calcium)) / np.std(calcium)
    isosbestic = (isosbestic - np.median(isosbestic)) / np.std(isosbestic)
    m = np.polyfit(isosbestic, calcium, 1)
    ref = isosbestic * m[0] + m[1]
    ph = (calcium - ref) / 100
    return ph

def preprocessing_alejandro(f_ca, fs, window=30):
    # https://www.biorxiv.org/content/10.1101/2024.02.26.582199v1
    """
    Fluorescence signals recorded during each session from each location were
    transformed to dF/F using the following formula: dF = (F-F0)/F0
    𝐹0 was the +/- 30 s rolling average of the raw fluorescence signal.
    """
    # Convert to Series to apply the rolling avg
    f_ca = pd.Series(f_ca)
    f0 = f_ca.rolling(int(fs * window), center=True).mean()
    delta_f = (f_ca - f0) / f0
    # Convert to numpy for output
    delta_f = delta_f.to_numpy()
    return delta_f

""" previously used functions """
# df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
# df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
# df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
# df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
# df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
# df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)
# df_nph['calcium_alex'] = preprocessing_alejandro(df_nph["raw_calcium"], fs=fs) 
# df_nph['isos_alex'] = preprocessing_alejandro(df_nph['raw_isosbestic'], fs=fs) 



# current code in the iblphotometry preprocessing file #the rest of the functions were removed 
# https://github.com/int-brain-lab/ibl-photometry/blob/f6f479a479ce327e6ba485ca449b19299795a86b/src/iblphotometry/preprocessing.py

import scipy.signal


def low_pass_filter(raw_signal, fs):
    params = {}
    sos = scipy.signal.butter(
        fs=fs,
        output='sos',
        **params.get('butterworth_lowpass', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}),
    )
    signal_lp = scipy.signal.sosfiltfilt(sos, raw_signal)
    return signal_lp


def mad_raw_signal(raw_signal, fs):
    # This is a convenience function to get going whilst the preprocessing refactoring is being done
    # TODO delete this function once processing can be applied
    signal_lp = low_pass_filter(raw_signal, fs)
    signal_processed = (raw_signal - signal_lp) / signal_lp
    return signal_processed

#%%
from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd

def preprocess_photometry(
    df_nph,
    calcium_col="raw_calcium",
    isosbestic_col="raw_isosbestic",
    fs=None,
    lowpass_hz=2,
    zscore=True
):
    """
    Preprocess photometry data by:
    1) Lowpass filtering.
    2) Regressing out isosbestic (motion/bleaching correction).
    3) Computing ΔF/F.
    4) Optionally z-scoring.

    Parameters:
    - df_nph: DataFrame with columns ['times', raw_calcium, raw_isosbestic].
    - calcium_col: Name of the calcium channel column.
    - isosbestic_col: Name of the isosbestic channel column.
    - fs: Sampling rate (Hz). If None, will estimate from 'times'.
    - lowpass_hz: Lowpass cutoff frequency (Hz).
    - zscore: Whether to z-score the ΔF/F signal.

    Returns:
    - df_out: DataFrame with ['times', 'dff', 'dff_zscore']
    """

    times = df_nph["times"].values
    raw_calcium = df_nph[calcium_col].values
    raw_isosbestic = df_nph[isosbestic_col].values

    # 1) Estimate sampling rate if not given
    if fs is None:
        fs = 1.0 / np.median(np.diff(times))
        print(f"Estimated sampling rate: {fs:.2f} Hz")

    # 2) Lowpass filter both channels
    b, a = butter(2, lowpass_hz / (fs / 2), btype="low")
    calcium_filt = filtfilt(b, a, raw_calcium)
    isosbestic_filt = filtfilt(b, a, raw_isosbestic)

    # 3) Regress isosbestic out of calcium
    # Fit: calcium_filt = beta * isosbestic_filt + intercept
    A = np.vstack([isosbestic_filt, np.ones_like(isosbestic_filt)]).T
    beta, intercept = np.linalg.lstsq(A, calcium_filt, rcond=None)[0]
    fitted_isosbestic = beta * isosbestic_filt + intercept

    corrected = calcium_filt - fitted_isosbestic

    # 4) Compute ΔF/F
    baseline = np.median(fitted_isosbestic)
    dff = corrected / baseline

    # 5) Z-score if requested
    if zscore:
        dff_zscore = (dff - np.mean(dff)) / np.std(dff)
    else:
        dff_zscore = dff

    # 6) Prepare output DataFrame
    df_out = pd.DataFrame({
        "times": times,
        "dff": dff,
        "dff_zscore": dff_zscore
    })

    return df_out









'''
get_zdFF.py calculates standardized dF/F signal based on calcium-idependent 
and calcium-dependent signals commonly recorded using fiber photometry calcium imaging

Ocober 2019 Ekaterina Martianova ekaterina.martianova.1@ulaval.ca 

Reference:
  (1) Martianova, E., Aronson, S., Proulx, C.D. Multi-Fiber Photometry 
      to Record Neural Activity in Freely Moving Animal. J. Vis. Exp. 
      (152), e60278, doi:10.3791/60278 (2019)
      https://www.jove.com/video/60278/multi-fiber-photometry-to-record-neural-activity-freely-moving

'''

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

import numpy as np
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







import numpy as np

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
#found in kcenia branch in github iblphotometry

"""
This modules offers pre-processing for raw photometry data.
It implements different kinds of pre-processings depending on the preference of the user.
Where applicable, I have indicated a publication with the methods.
"""
import scipy.signal
import numpy as np

import ibldsp.utils
from iblutil.numerical import rcoeff


def preprocess_sliding_mad(raw_calcium, times, fs=None, wlen=120, overlap=90, returns_gain=False, **params):
    """
    Applies one pass of fiber photobleaching
    :param raw_calcium:
    :param times:
    :param fs:
    :param wlen:
    :param overlap:
    :param params:
    :return:
    """
    calcium = photobleaching_lowpass(raw_calcium, fs, **params)
    wg = ibldsp.utils.WindowGenerator(ns=calcium.size, nswin=int(wlen * fs), overlap=overlap)
    trms = np.array([first for first, last in wg.firstlast]) / fs + times[0]
    rmswin, _ = psth(calcium, times, t_events=trms, fs=fs, peri_event_window=[0, wlen])
    gain = np.nanmedian(np.abs(calcium)) / np.nanmedian(np.abs(rmswin), axis=0)
    gain = np.interp(times, trms, gain)
    if returns_gain:
        return calcium * gain, gain
    else:
        return calcium * gain


def jove2019(raw_calcium, raw_isosbestic, fs, **params):
    """
    Martianova, Ekaterina, Sage Aronson, and Christophe D. Proulx. "Multi-fiber photometry to record neural activity in freely-moving animals." JoVE (Journal of Visualized Experiments) 152 (2019): e60278.
    :param raw_calcium:
    :param raw_isosbestic:
    :param params:
    :return:
    """
    # the first step is to remove the photobleaching w
    sos = scipy.signal.butter(fs=fs, output='sos', **params.get('butterworth_lowpass', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}))
    calcium = raw_calcium - scipy.signal.sosfiltfilt(sos, raw_calcium)
    isosbestic = raw_isosbestic - scipy.signal.sosfiltfilt(sos, raw_isosbestic)
    calcium = (calcium - np.median(calcium)) / np.std(calcium)
    isosbestic = (isosbestic - np.median(isosbestic)) / np.std(isosbestic)
    m = np.polyfit(isosbestic, calcium, 1)
    ref = isosbestic * m[0] + m[1]
    ph = (calcium - ref) / 100
    return ph


def photobleaching_lowpass(raw_calcium, fs, **params):
    """
    Here the isosbestic recording is ignored, and the reference is computed as the low-pass component of the
    calcium band signal.
    :param calcium:
    :param params: dictionary with parameters
    {
        'butterworth_lowpass': {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}
        }
        dictionary with parameters for the butterworth filter applied to the calcium band for the sole purpose of regression
    :return:
    iso (np.array): the corrected isosbestic signal to be used as control
    ph (np.array): the corrected calcium signal
    """
    params = {} if params is None else params
    sos = scipy.signal.butter(fs=fs, output='sos', **params.get('butterworth_lowpass', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}))
    calcium_lp = scipy.signal.sosfiltfilt(sos, raw_calcium)
    calcium = (raw_calcium - calcium_lp) / calcium_lp
    return calcium


def isosbestic_regression(raw_isosbestic, raw_calcium, fs, **params):
    """
    Prototype of baseline correction for photometry data.
    Fits a low pass version of the isosbestic signal to the calcium signal. The baseline signal is
    the low pass isosbestic signal multiplied by the fit slope and added to the fit intercept.
    The corrected signal is the calcium signal minus the baseline signal divided by the baseline signal.
    We apply the same procedure to the full-band isosbestic signal to check for remaining correlations.
    :param raw_isosbestic:
    :param raw_calcium:
    :param params: dictionary with parameters
        butterworth_regression: dictionary with parameters for the butterworth filter applied to both isosbestic and
        calcium band for the sole purpose of regression {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}
        butterworth_signal: dictionary with parameters for the butterworth filter {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}
        applied to the outputs. Set to None to disable filtering
    :return:
    iso (np.array): the corrected isosbestic signal to be used as control
    ph (np.array): the corrected calcium signal
    """
    params = {} if params is None else params

    sos = scipy.signal.butter(**params.get('butterworth_regression', {'N': 3, 'Wn': 0.1, 'btype': 'lowpass', 'fs': fs}), output='sos')
    calcium_lp = scipy.signal.sosfiltfilt(sos, raw_calcium)
    isosbestic_lp = scipy.signal.sosfiltfilt(sos, raw_isosbestic)
    m = np.polyfit(isosbestic_lp, calcium_lp, 1)

    ref = isosbestic_lp * m[0] + m[1]
    ph = (raw_calcium - ref) / ref

    butterworth_signal = params.get('butterworth_signal', {'N': 3, 'Wn': 10, 'btype': 'lowpass', 'fs': fs})
    if butterworth_signal is not None:
        sosbp = scipy.signal.butter(**butterworth_signal, output='sos')
        ph = scipy.signal.sosfiltfilt(sosbp, ph)
    return ph


def isosbestic_correction_dataframe(df_photometry):
    """
    Wrapper around the baseline correction function to apply it to a dataframe with the raw signals
    `calcium` is the corrected calcium signal
    `isosbestic_control` is the isosbestic signal having gone through the same correction procedure
    :param df_photometry: should contain columns `raw_isosbestic' and `raw_calcium'
    :return: df_photometry with columns `calcium' and `isosbestic_control'
    """
    fs = 1 / np.median(np.diff(df_photometry['times'].values))
    ph = isosbestic_regression(df_photometry['raw_isosbestic'].values, df_photometry['raw_calcium'].values, fs=fs)
    iso = isosbestic_regression(df_photometry['raw_isosbestic'].values, df_photometry['raw_isosbestic'].values, fs=fs)
    df_photometry['isosbestic_control'] = iso
    df_photometry['calcium'] = ph
    return df_photometry


def psth(calcium, times, t_events, fs=None, peri_event_window=None):
    """
    Compute the peri-event time histogram of a calcium signal
    :param calcium:
    :param times:
    :param t_events:
    :param fs:
    :param peri_event_window:
    :return:
    """
    fs = 1 / np.median(np.diff(times)) if fs is None else fs
    peri_event_window = [-1, 2] if peri_event_window is None else peri_event_window
    # compute a vector of indices corresponding to the perievent window at the given sampling rate
    sample_window = np.round(np.arange(peri_event_window[0] * fs, peri_event_window[1] * fs + 1)).astype(int)
    # we inflate this vector to a 2d array where each column corresponds to an event
    idx_psth = np.tile(sample_window[:, np.newaxis], (1, t_events.size))
    # we add the index of each event too their respective column
    idx_event = np.searchsorted(times, t_events)
    idx_psth += idx_event
    i_out_of_bounds = np.logical_or(idx_psth > (calcium.size - 1), idx_psth < 0)
    idx_psth[i_out_of_bounds] = -1
    psth = calcium[idx_psth]  # psth is a 2d array (ntimes, nevents)
    psth[i_out_of_bounds] = np.nan  # remove events that are out of bounds
    return psth, idx_psth


def sliding_rcoeff(signal_a, signal_b, nswin, overlap=0):
    """
    Computes the local correlation coefficient between two signals in sliding windows
    :param signal_a:
    :param signal_b:
    :param nswin: window size in samples
    :param overlap: overlap of successiv windows in samples
    :return: ix: indices of the center of the windows, r: correlation coefficients
    """
    wg = ibldsp.utils.WindowGenerator(ns=signal_a.size, nswin=nswin, overlap=overlap)
    first_samples = np.array([fl[0] for fl in wg.firstlast])
    iwin = np.zeros([wg.nwin, wg.nswin], dtype=np.int32) + np.arange(wg.nswin)
    iwin += first_samples[:, np.newaxis]
    iwin[iwin >= signal_a.size] = signal_a.size - 1
    r = rcoeff(signal_a[iwin], signal_b[iwin])
    ix = first_samples + nswin // 2
    return ix, r