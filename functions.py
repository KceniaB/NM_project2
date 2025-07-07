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


""" LOAD TRIALS """
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
