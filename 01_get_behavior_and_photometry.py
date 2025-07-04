# %%
""" 
Load behavior and photometry data 
KB - 2025-July-04 

"""
# %%
'/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/data/external_drive' 

#%%
"""
KceniaBougrova 
08October2024 

1. LOAD THE BEHAVIOR AND PHOTOMETRY FILES
2. ADD BEHAVIOR VARIABLES 
3. SYNCHRONIZE BEHAV AND PHOTOMETRY 
4. PREPROCESS PHOTOMETRY
5. PLOT HEATMAP AND LINEPLOT DIVIDED BY FEEDBACK TYPE 

""" 

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from brainbox.behavior.training import compute_performance 
from brainbox.io.one import SessionLoader 
# import iblphotometry.kcenia as kcenia
import ibldsp.utils
import scipy.signal
from iblutil.numerical import rcoeff
from one.api import ONE #always after the imports 
# one = ONE(cache_dir="/mnt/h0/kb/data/one") 
one = ONE() 

""" useful""" 
# eids = one.search(project='ibl_fibrephotometry') 

#%%
""" EDIT THE VARS - eid, ROI, photometry file path (.csv or .pqt) """
eid = '56ed83ac-c196-4817-bc37-62c02ba89d47' #example eid 

#choose one: .csv or .pqt
nph_file_path = '/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/data/one/mainenlab/Subjects/ZFM-06305/2023-08-31/001/raw_photometry_data/raw_photometry.csv' 
nph_bnc_path = '/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/data/external_drive/2023-08-31/bonsai_DI10.csv'
df_nph = pd.read_csv(nph_file_path) 
df_bnc = pd.read_csv(nph_bnc_path) 
# df_nph = pd.read_parquet(nph_file_path) 

region_number = 3
region_number = 4

#%%
""" LOAD TRIALS """
def load_trials_updated(eid=eid): 
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

df_trials, subject, session_date = load_trials_updated(eid) 

#%% #########################################################################################################
""" GET PHOTOMETRY DATA """ 

region = f"Region{region_number}G"

df_nph["mouse"] = subject
df_nph["date"] = session_date
df_nph["region"] = region
df_nph["eid"] = eid 


"""
CHANGE INPUT AUTOMATICALLY 
""" 
iup = ibldsp.utils.rises(df_nph[f'Input{nph_bnc}'].values) #idx nph TTL times 
tph = (df_nph['Timestamp'].values[iup] + df_nph['Timestamp'].values[iup - 1]) / 2 #nph TTL times computed for the midvalue 
tbpod = np.sort(np.r_[
    df_trials['intervals_0'].values,
    df_trials['intervals_1'].values - 1,  # here is the trick
    df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
)
fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)

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
df_nph = kcenia.LedState_or_Flags(df_nph)

""" 4.1.2 Check for LedState/previous Flags bugs """ 
""" 4.1.2.1 Length """
# Verify the length of the data of the 2 different LEDs
df_470, df_415 = kcenia.verify_length(df_nph)
""" 4.1.2.2 Verify if there are repeated flags """ 
kcenia.verify_repetitions(df_nph["LedState"])
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

my_array = np.c_[raw_timestamps_bpod, raw_reference, raw_signal]

df_nph = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium']) #IMPORTANT DF

from iblphotometry.preprocessing import preprocessing_alejandro, jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 

# def jove2019(raw_calcium, raw_isosbestic, fs, **params):
#     """
#     Martianova, Ekaterina, Sage Aronson, and Christophe D. Proulx. "Multi-fiber photometry to record neural activity in freely-moving animals." JoVE (Journal of Visualized Experiments) 152 (2019): e60278.
#     :param raw_calcium:
#     :param raw_isosbestic:
#     :param params:
#     :return:
#     """
#     # the first step is to remove the photobleaching w
#     sos = scipy.signal.butter(fs=fs, output='sos', **params.get('butterworth_lowpass', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}))
#     calcium = raw_calcium - scipy.signal.sosfiltfilt(sos, raw_calcium)
#     isosbestic = raw_isosbestic - scipy.signal.sosfiltfilt(sos, raw_isosbestic)
#     calcium = (calcium - np.median(calcium)) / np.std(calcium)
#     isosbestic = (isosbestic - np.median(isosbestic)) / np.std(isosbestic)
#     m = np.polyfit(isosbestic, calcium, 1)
#     ref = isosbestic * m[0] + m[1]
#     ph = (calcium - ref) / 100
#     return ph

# def preprocessing_alejandro(f_ca, fs, window=30):
#     # https://www.biorxiv.org/content/10.1101/2024.02.26.582199v1
#     """
#     Fluorescence signals recorded during each session from each location were
#     transformed to dF/F using the following formula: dF = (F-F0)/F0
#     𝐹0 was the +/- 30 s rolling average of the raw fluorescence signal.
#     """
#     # Convert to Series to apply the rolling avg
#     f_ca = pd.Series(f_ca)
#     f0 = f_ca.rolling(int(fs * window), center=True).mean()
#     delta_f = (f_ca - f0) / f0
#     # Convert to numpy for output
#     delta_f = delta_f.to_numpy()
#     return delta_f


# df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
# df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
# df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
# df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)
# df_nph['calcium_alex'] = preprocessing_alejandro(df_nph["raw_calcium"], fs=fs) 
# df_nph['isos_alex'] = preprocessing_alejandro(df_nph['raw_isosbestic'], fs=fs)

plt.figure(figsize=(20, 6))
plt.plot(df_nph['times'][200:1000], df_nph['calcium_mad'][200:1000], linewidth=1.25, alpha=0.8, color='teal') 
plt.plot(df_nph['times'][200:1000], df_nph['isosbestic_mad'][200:1000], linewidth=1.25, alpha=0.8, color='purple') 
plt.show() 

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


# photometry_s_1 = df_nph.calcium_photobleach.values[psth_idx] 
# # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_1)
# photometry_s_2 = df_nph.isosbestic_photobleach.values[psth_idx] 
# # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_2)
# photometry_s_3 = df_nph.calcium_jove2019.values[psth_idx] 
# # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_3)
# photometry_s_4 = df_nph.isosbestic_jove2019.values[psth_idx] 
# # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_4)
photometry_s_5 = df_nph.calcium_mad.values[psth_idx] 
# np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_5)
photometry_s_6 = df_nph.isosbestic_mad.values[psth_idx] 
# # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_6) 
# photometry_s_7 = df_nph.calcium_alex.values[psth_idx] 
# photometry_s_8 = df_nph.isos_alex.values[psth_idx] 

def plot_heatmap_psth(preprocessingtype=df_nph.calcium_mad): 
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

    fig.suptitle(f'calcium_mad_{EVENT}_{subject}_{session_date}_{region}_{eid}', y=1, fontsize=14)
    plt.tight_layout()
    # plt.savefig(f'/mnt/h0/kb/data/psth_npy/Fig02_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
    plt.show() 

plot_heatmap_psth(df_nph.calcium_mad) 


