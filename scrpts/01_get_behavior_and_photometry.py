# %%
""" 
Load behavior and photometry data 
KB - 2025-July-04 

"""
# %%
'/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/data/external_drive' 

'/home/kceniabougrova/Documents/NM_project_fromIBLserver/NM_project2/KB_sessions_insertions_map - upload.csv'
#%%
"""
KceniaBougrova 
08October2024
30July2025 updated/optimized

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
from functions import *
import sys
sys.path.insert(0, "/home/kceniabougrova/Documents/GitHub/ibl-photometry/src")

from one.api import ONE #always after the imports 
# one = ONE(cache_dir="/mnt/h0/kb/data/one") 
one = ONE() 

""" useful""" 
# eids = one.search(project='ibl_fibrephotometry') 

#%%
#===========================================================================
#                            Pick data from table 
#===========================================================================

table_path = '/home/kceniabougrova/Documents/NM_project_fromIBLserver/NM_project2/KB_sessions_insertions_map - upload.csv'
sessions_list = pd.read_csv(table_path)

def select_one_session(sessions_list=sessions_list, row=150): 
    eid = sessions_list.loc[row, "eid"]
    subject = sessions_list.loc[row, "subject"]
    date = sessions_list.loc[row, "date"]
    region = sessions_list.loc[row, "region"]
    nph_file_path = sessions_list.loc[row, "photometry_path_a"]
    nph_bnc_path = sessions_list.loc[row, "digital_inputs_path"]
    return eid, subject, date, region, nph_file_path, nph_bnc_path


eid, subject, date, region, nph_file_path, nph_bnc_path = select_one_session(row=450)
# Replace the prefix in the paths if needed
old_prefix = '/mnt/h0/kb/'
new_prefix = '/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/'

nph_file_path = nph_file_path.replace(old_prefix, new_prefix)
nph_bnc_path = nph_bnc_path.replace(old_prefix, new_prefix)

# Load the CSVs
df_nph = pd.read_csv(nph_file_path)
df_bnc = pd.read_csv(nph_bnc_path)



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




alldata = pd.read_csv('/home/kceniabougrova/Downloads/KB_sessions_insertions_map - sessions_table(1).csv')

#%%


df_trials, subject, session_date = load_trials_updated(eid) 

#%% #########################################################################################################
""" GET PHOTOMETRY DATA """ 

region = f"Region{region_number}G"

df_nph["mouse"] = subject
df_nph["date"] = session_date
df_nph["region"] = region
df_nph["eid"] = eid 
nph_bnc = df_bnc

# Remove 'Value.' prefix from columns
nph_bnc.columns = [col.replace("Value.", "") for col in nph_bnc.columns]
nph_bnc = nph_bnc[nph_bnc["Value"] == True].reset_index(drop=True)

tph = df_bnc["Timestamp"].values


"""
CHANGE INPUT AUTOMATICALLY 
""" 
# tph = (df_nph['Timestamp'].values[iup] + df_nph['Timestamp'].values[iup - 1]) / 2 #nph TTL times computed for the midvalue 
tbpod = np.sort(np.r_[
    df_trials['intervals_0'].values,
    df_trials['intervals_1'].values - 1,  # here is the trick
    df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
)

fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)

df_nph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_nph["Timestamp"]) 

fcn_nph_to_bpod_times(df_nph["Timestamp"])


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
plt.plot(df_nph['times'][200:1000], df_nph['raw_calcium'][200:1000], linewidth=1.25, alpha=0.8, color='teal') 
plt.plot(df_nph['times'][200:1000], df_nph['raw_isosbestic'][200:1000], linewidth=1.25, alpha=0.8, color='purple') 
plt.show() 

import pandas as pd
import matplotlib.pyplot as plt









# ==========================================
# 1) Preprocess over the entire continuous signal
# ==========================================
df_global = preprocess_photometry(df_nph)

# ==========================================
# 2) Preprocess per trial
# ==========================================
#####
""" but first, include a column with the trial number - 30072025"""
""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
df_nph.loc[idx_event,"trial_number"]=1
df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

#####
dfs = []
for trial, df_trial in df_nph.groupby("trial_number"):
    df_clean = preprocess_photometry(df_trial)
    df_clean["trial_number"] = trial
    dfs.append(df_clean)

df_per_trial = pd.concat(dfs, ignore_index=True)

# ==========================================
# 3) Prepare raw signal
# ==========================================
times = df_nph["times"].values
raw_calcium = df_nph["raw_calcium"].values

# ==========================================
# 4) Define event times (tbpod)
# ==========================================
# done before 

# ==========================================
# 5) Plot comparison
# ==========================================
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# --- Raw signal ---
axs[0].plot(times, raw_calcium, color="gray", alpha=0.7)
axs[0].set_title("Raw Calcium Signal")
axs[0].set_ylabel("Raw Fluorescence")

# --- Global preprocessing ---
axs[1].plot(df_global["times"], df_global["dff_zscore"], color="blue", alpha=0.7)
axs[1].set_title("Global Preprocessing (entire signal)")
axs[1].set_ylabel("ΔF/F (z-score)")

# --- Per-trial preprocessing ---
axs[2].plot(df_per_trial["times"], df_per_trial["dff_zscore"], color="green", alpha=0.7)
axs[2].set_title("Per-Trial Preprocessing")
axs[2].set_ylabel("ΔF/F (z-score)")
axs[2].set_xlabel("Time (s)")

# --- Add vertical lines for event times ---
for ax in axs:
    for t in tbpod:
        ax.axvline(t, color="black", alpha=0.5, linewidth=0.5)
plt.xlim(1510,1550)
plt.tight_layout()
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


photometry_s_1 = df_global["dff_zscore"].values[psth_idx]
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



# %%
#===========================================================================
#                            Photometry done
#                       now link it to the behavior 
#===========================================================================


PERIEVENT_WINDOW = [-1, 2]
EVENT = "feedback_times"
#load data 
behav = pd.read_parquet('/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-05236/2023-07-03/001/alf/_ibl_trials.table.pqt') 


#preprocess data 
fs = 1 / np.median(np.diff(nph.times.values))
nph_j = jove2019(nph.raw_calcium, nph.raw_isosbestic, fs=fs) 
# nph_j = preprocess_sliding_mad(nph.raw_calcium.values, nph.times.values, fs=fs)
nph["calcium"] = nph_j 

plt.figure(figsize=(20, 8))
plt.plot(nph.times, nph.calcium, c='teal', alpha=0.85, linewidth=0.2)
for i in behav.feedback_times: 
    plt.axvline(x=i, linewidth=0.2, color='black', alpha=0.75) 
plt.show() 


#create psth #OPTION1 

photometry_feedback, idx_psth = psth(
    calcium=nph.calcium.values,
    times=nph.times.values, t_events=behav[EVENT].values, fs=fs, peri_event_window =PERIEVENT_WINDOW) 

plt.figure(figsize=(15, 8))
plt.plot(photometry_feedback, color='black', linewidth=0.3, alpha=0.3) 
plt.axvline(x=30)
plt.show()

#check peak, time, mad 

ipeaks_1, maxis = parabolic_max(photometry_feedback.T)
tmax = ipeaks_1 / fs + PERIEVENT_WINDOW[0]
plt.figure(figsize=(10, 5))

plt.matshow(photometry_feedback)
plt.plot(np.arange(ipeaks_1.shape[0]), ipeaks_1, '*r')
def mad(arr, axis=None, keepdims=True):
    median = np.median(arr, axis=axis, keepdims=True)
    mad = np.median(np.abs(arr-median),axis=axis, keepdims=keepdims)
    return mad
ipeaks=[] 
mads=[]
for i in range(len(photometry_feedback)): 
    ipeaks.append(np.argmax(photometry_feedback[i])) #indices of max values along axis
    mads.append(mad(photometry_feedback[i]))



#%% 
"""
KB 20240709 
adding some more behav columns 
""" 
#createtrialNumber
behav['trialNumber'] = range(1, len(behav) + 1)
idx=2 #column position 
new_col = behav['contrastLeft'].fillna(behav['contrastRight']) 
behav.insert(loc=idx, column='allContrasts', value=new_col) 
#create allUContrasts 
behav['allUContrasts'] = behav['allContrasts']
behav.loc[behav['contrastRight'].isna(), 'allUContrasts'] = behav['allContrasts'] * -1
behav.insert(loc=3, column='allUContrasts', value=behav.pop('allUContrasts'))
#create reactionTime 
reactionTime = np.array((behav["firstMovement_times"])-(behav["stimOn_times"]))
behav["reactionTime"] = reactionTime 

event_time = 30 

psth_fastest_100 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 1))]]
psth_fastest_25 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 0.25))]]
psth_fastest_12 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 0.125))]]
psth_fastest_6 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 0.0625))]]
psth_fastest_0 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 0))]]
# psth_fastest_100 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 1))]]
# psth_fastest_25 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.25))]]
# psth_fastest_12 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.125))]]
# psth_fastest_6 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.0625))]]
# psth_fastest_0 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0))]]



width_ratios = [1, 1]
FONTSIZE_1 = 30
FONTSIZE_2 = 25
FONTSIZE_3 = 15 

data = nph.calcium.values[idx_psth]
event_time = 30 

from numpy import nanmean
average_values = nanmean(data,axis=1)

plt.plot(average_values, color='black')
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
plt.grid(False) 


psth_error = nph.calcium.values[idx_psth[:,(behav.feedbackType == -1)]]
psth_good = nph.calcium.values[idx_psth[:,(behav.feedbackType == 1)]]

"""#########################################################################################################################################"""
""" WORKED 
Plot heatmap for correct and incorrect 
"""
psth_good_avg = psth_good.mean(axis=1) 
sem_A = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1]) 

ax = sns.heatmap(psth_good.T, cbar=True)
ax.invert_yaxis()
plt.axvline(x=30, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed") 
plt.title("Heatmap for correct trials")
plt.show()
 
ax = sns.heatmap(psth_error.T, cbar=True)
ax.invert_yaxis()
plt.axvline(x=30, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed") 
plt.title("Heatmap for incorrect trials")
plt.show() 






"""#########################################################################################################################################"""
"""##### scatter of max and min peaks divided by correct and incorrect ##########################################################"""
""" WORKED
Figure X1. where is the max peak around the event? 
"""
ipeaks_1, maxis = parabolic_max(psth_good.T)
tmax = ipeaks_1 / fs + PERIEVENT_WINDOW[0]

ipeaks_2, maxis_2 = parabolic_max(psth_error.T)
tmax_2 = ipeaks_2 / fs + PERIEVENT_WINDOW[0]

import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.scatter(tmax, maxis, color='#0f7173', alpha=0.5, label='correct') 
plt.scatter(tmax_2, maxis_2, color='#f05d5e', alpha=0.5, label='incorrect')
plt.axvline(x=0, linestyle='dashed', color='black')

plt.title("max peak values divided by correct and incorrect")
plt.legend()
plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
""" WORKED 
Figure X2. where is the min peak around the event? 
""" 
ipeaks_3, minis = parabolic_max(-(psth_good.T))
tmin = ipeaks_3 / fs + PERIEVENT_WINDOW[0]

ipeaks_4, minis_2 = parabolic_max(-(psth_error.T))
tmin_2 = ipeaks_4 / fs + PERIEVENT_WINDOW[0]

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.scatter(tmin, minis, color='#0f7173', alpha=0.5, label='correct') 
plt.scatter(tmin_2, minis_2, color='#f05d5e', alpha=0.5, label='incorrect')
plt.axvline(x=0, linestyle='dashed', color='black') 

plt.title("min peak values divided by correct and incorrect")
plt.legend()
plt.show() 

