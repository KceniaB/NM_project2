#%% 
""" 
New preprocessing - all in this repo 
"""
# %%
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


table_path = '/home/kceniabougrova/Downloads/NewSessions_Aug2025_New.csv' 
sessions_list = pd.read_csv(table_path)

def get_eid(mouse, date): 
    eids = one.search(subject=mouse, date=date) 
    if len(eids) == 0:
        return None  # no session found
    eid = str(eids[0])   # get the first one and make it a string
    ref = one.eid2ref(eid)
    print(eid)
    print(ref) 
    return eid


def select_one_session(sessions_list, row=12):
    subject = sessions_list.loc[row, "subject"]
    date = sessions_list.loc[row, "date"]
    region = sessions_list.loc[row, "region"]
    nph_file_path = sessions_list.loc[row, "photometry_path_a"]
    nph_bnc_path = sessions_list.loc[row, "digital_inputs_path"]
    eid = get_eid(subject,date)
    print(eid)
    return eid, subject, date, region, nph_file_path, nph_bnc_path


eid, subject, date, region, nph_file_path, nph_bnc_path = select_one_session(
    sessions_list, row=195)


old_prefix = '/mnt/h0/kb/'
new_prefix = '/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/'

nph_file_path = nph_file_path.replace(old_prefix, new_prefix)
nph_bnc_path = nph_bnc_path.replace(old_prefix, new_prefix)

# Load the CSVs
df_nph = pd.read_csv(nph_file_path)
df_bnc = pd.read_csv(nph_bnc_path)

# Load the behavior
df_trials, subject, session_date = load_trials_updated(eid) 

print(subject, date, region)


df_nph["mouse"] = subject
df_nph["date"] = session_date
df_nph["region"] = region
df_nph["eid"] = eid 

# Remove 'Value.' prefix from columns
df_bnc.columns = [col.replace("Value.", "") for col in df_bnc.columns]
df_bnc = df_bnc[df_bnc["Value"] == True].reset_index(drop=True)

tph = df_bnc["Timestamp"].values 



##############################################################################
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




print("Len TTL: ", len(tph), "Len tbpod: ", len(tbpod), "Len trials: ", len(df_trials))





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

# Pick the right column name automatically
colname = "LedState" if "LedState" in df_ph_1.columns else "Flags"

"""
##TO LEAD WITH REPEATED FLAGS 
# index where the swap happens
swap_idx = 9546  

# Make a copy so we don’t overwrite original
df_fixed = df_ph_1.copy()

# Flip flags from swap_idx onward
df_fixed.loc[swap_idx:, colname] = df_fixed.loc[swap_idx:, colname].map({1: 2, 2: 1})

df_ph_1 = df_fixed

"""

# Remove rows with col == 1 at both ends
if df_ph_1[colname].iloc[0] == 1 and df_ph_1[colname].iloc[-1] == 1:
    df_ph_1 = df_ph_1.iloc[1:]

# Remove rows with col == 2 at both ends
if df_ph_1[colname].iloc[0] == 2 and df_ph_1[colname].iloc[-1] == 2:
    df_ph_1 = df_ph_1.iloc[:-2]

# Filter data for LedState 2 (470nm)
df_470 = df_ph_1[df_ph_1[colname] == 2]

# Filter data for LedState 1 (415nm)
df_415 = df_ph_1[df_ph_1[colname] == 1]

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






##############################################################################################################################


raw_reference = df_nph['raw_isosbestic'][0:]
raw_signal = df_nph['raw_calcium'][0:]

smooth_win = 10
smooth_reference = smooth_signal(raw_reference, smooth_win)
smooth_signal = smooth_signal(raw_signal, smooth_win) 

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(smooth_signal,'blue',linewidth=0.5)
ax2 = fig.add_subplot(212)
ax2.plot(smooth_reference,'purple',linewidth=0.5)



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





remove=0
reference = (smooth_reference[remove:] - r_base[remove:])
signal = (smooth_signal[remove:] - s_base[remove:])  

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(signal,'blue',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(reference,'purple',linewidth=1.5)







z_reference = (reference - np.median(reference)) / np.std(reference)
z_signal = (signal - np.median(signal)) / np.std(signal)

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(z_signal,'blue',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(z_reference,'purple',linewidth=1.5)



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




fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(z_signal,'blue')
ax1.plot(z_reference_fitted,'purple')





zdFF = (z_signal - z_reference_fitted)



fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(zdFF,'black')


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
    t_events=behav["feedback_times"].values,
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

# time_axis = np.arange(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], 1/fs)
n_timepoints = photometry_feedback.shape[0]
time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

plt.figure(figsize=(15, 8))
plt.plot(time_axis, photometry_feedback, color='black', linewidth=0.3, alpha=0.3)
plt.axvline(x=0, color='red', linestyle='--')  # Event at 0s
plt.xlabel("Time (s)")






""" SELECT THE EVENT AND WHAT INTERVAL TO PLOT IN THE PSTH """ 
EVENT = "stimOnTrigger_times" 
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

plt.figure(figsize=(15, 8))

# Mask for correct vs incorrect/other
mask_correct = df_trials.feedbackType.values == 1
mask_incorrect = ~mask_correct

# Plot trials separately
plt.plot(time_axis, photometry_feedback[:, mask_correct], color='blue', linewidth=0.3, alpha=0.3)
plt.plot(time_axis, photometry_feedback[:, mask_incorrect], color='red', linewidth=0.3, alpha=0.3)

# Event marker
plt.axvline(x=0, color='black', linestyle='--')

plt.xlabel("Time (s)")
plt.ylabel("ΔF/F (z-scored)")
plt.title("Peri-feedback PSTH split by feedback type")
plt.show()







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


print(subject, date, region)
print(eid)
print("Len TTL: ", len(tph), "Len tbpod: ", len(tbpod), "Len trials: ", len(df_trials))































































































# %%
""" 
to sort by: reactionTimes 
""" 
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
prediction for the vars that would better explain the drop in the 5-HT activity before the stimOnTrigger_times 
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
# df_trials['consecutive_correct'] = df_trials['correct'].astype(int).rolling(window=2).sum().fillna(0)

# Counter for consecutive corrects before current trial
consec = []
count = 0
for c in df_trials['correct']:
    consec.append(count)  # store count BEFORE updating
    if c:
        count += 1
    else:
        count = 0

df_trials['consecutive_correct'] = consec

# Add previous trial's quiescence time
df_trials['prev_quiescenceTime'] = df_trials['quiescenceTime'].shift(1)

# Add previous trial's full trial time
df_trials['prev_trialTime'] = df_trials['trialTime'].shift(1)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Choose predictors
X = df_trials[['quiescenceTime', 'reactionTime', 'consecutive_correct', 'prev_quiescenceTime', 'prev_trialTime']]
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
sns.scatterplot(data=df_trials, x='quiescenceTime', y='zdff_drop_pre_stim', hue='feedbackType', alpha=0.8)

sns.pairplot(df_trials, vars=['zdff_drop_pre_stim', 'quiescenceTime', 'reactionTime', 'consecutive_correct', 'prev_quiescenceTime', 'prev_trialTime'], hue='correct')


from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Fit Random Forest again
rf = RandomForestRegressor(random_state=42, n_estimators=300)
rf.fit(X_scaled, y)

# Plot PDP for all features
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(rf, X_scaled, [0, 1, 2, 3],
                                        feature_names=X.columns, ax=ax)
plt.suptitle("Partial Dependence Plots (Random Forest)", fontsize=14)
plt.show()


import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_scaled)

# Summary plot (global importance + direction of effect)
shap.summary_plot(shap_values, X, feature_names=X.columns)

# Dependence plot (feature vs its SHAP value)
shap.dependence_plot("quiescenceTime", shap_values, X, feature_names=X.columns)





#%%
# """
# 20August2025
# more features 
# """ 
# # %%
# """
# Pipeline to predict zdFF drop before stimOnTrigger_times
# with multiple models and extended feature set (A–D).
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error, r2_score

# # ==========================================================
# # 1. FEATURE ENGINEERING
# # ==========================================================

# # ---- Current trial (A)
# df_trials['quiescencePeriod'] = df_trials['quiescencePeriod']
# df_trials['quiescenceTime']   = df_trials['quiescenceTime']
# df_trials['trialTime']        = df_trials['trialTime']

# # ---- Previous trial (B)
# shift_cols = [
#     'quiescenceTime', 'trialTime', 'feedbackType', 'choice',
#     'allContrasts', 'allSContrasts', 'reactionTime',
#     'probabilityLeft', 'responseTime', 'zdff_drop_pre_stim'
# ]
# for col in shift_cols:
#     df_trials[f'prev_{col}'] = df_trials[col].shift(1)

# # ---- Sequential (C)
# df_trials['correct'] = (df_trials['feedbackType'] == 1).astype(int)

# # Consecutive correct
# consec_corr, count = [], 0
# for c in df_trials['correct']:
#     consec_corr.append(count)
#     count = count + 1 if c else 0
# df_trials['consecutive_correct'] = consec_corr

# # Consecutive incorrect
# consec_inc, count = [], 0
# for c in df_trials['correct']:
#     consec_inc.append(count)
#     count = count + 1 if not c else 0
# df_trials['consecutive_incorrect'] = consec_inc

# # ---- Interactions (D)
# df_trials['interaction_quiescence_correct'] = df_trials['quiescenceTime'] * df_trials['correct']
# df_trials['interaction_reaction_correct'] = df_trials['reactionTime'] * df_trials['correct']
# df_trials['interaction_prev_trialTime_quiescenceTime'] = df_trials['prev_trialTime'] * df_trials['quiescenceTime']
# df_trials['interaction_prev_feedbackType_quiescenceTime'] = df_trials['prev_feedbackType'] * df_trials['quiescenceTime']

# # ==========================================================
# # 2. DEFINE PREDICTORS + TARGET
# # ==========================================================

# # Collect all engineered features (exclude target + IDs)
# exclude_cols = ['zdff_drop_pre_stim', 'stimOnTrigger_times', 'subject', 'date','eid']
# feature_cols = [c for c in df_trials.columns if c not in exclude_cols]

# X = df_trials[feature_cols].fillna(0)
# y = df_trials['zdff_drop_pre_stim']

# # Drop NaNs in target
# mask = y.notna()
# X = X[mask]
# y = y[mask]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize for models that need scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ==========================================================
# # 3. TRAIN MULTIPLE MODELS
# # ==========================================================

# models = {
#     "LinearRegression": LinearRegression(),
#     "LassoCV": LassoCV(cv=5, random_state=42),
#     "ElasticNetCV": ElasticNetCV(cv=5, random_state=42),
#     "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
#     "GradientBoosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
#     "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1)
# }

# results = {}

# for name, model in models.items():
#     if name in ["LinearRegression", "LassoCV", "ElasticNetCV", "SVR"]:
#         model.fit(X_train_scaled, y_train)
#         y_pred = model.predict(X_test_scaled)
#     else:
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
    
#     r2 = r2_score(y_test, y_pred)
#     rmse = mean_squared_error(y_test, y_pred, squared=False)
#     results[name] = {"R2": r2, "RMSE": rmse}
#     print(f"=== {name} ===")
#     print(f"R²: {r2:.3f} | RMSE: {rmse:.3f}")
#     print()

# # ==========================================================
# # 4. FEATURE IMPORTANCES / COEFFICIENTS
# # ==========================================================

# # Linear models: coefficients
# for name in ["LinearRegression", "LassoCV", "ElasticNetCV"]:
#     model = models[name]
#     if hasattr(model, "coef_"):
#         coeffs = pd.Series(model.coef_, index=X.columns)
#         print(f"\n{name} coefficients (top 10):")
#         print(coeffs.reindex(coeffs.abs().sort_values(ascending=False).index)[:10])

# # Tree models: feature importances
# for name in ["RandomForest", "GradientBoosting"]:
#     model = models[name]
#     importances = pd.Series(model.feature_importances_, index=X.columns)
#     importances = importances.sort_values(ascending=False)[:15]
#     plt.figure(figsize=(8,6))
#     importances.plot(kind='barh')
#     plt.title(f"{name} - Top 15 Feature Importances")
#     plt.gca().invert_yaxis()
#     plt.show()








#%%
"""
20-August-2025
""" 
# %%
# %%
"""
21-August-2025 - EDITED, complete
Prediction of zdFF drop before stimOnTrigger_times
mean min slope
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor


import shap
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb

# ==========================================================
# 1. COMPUTE TARGETS (zdFF drop features per trial)
# ==========================================================

pre_window = [-0.5, 0]  # seconds relative to stim

means, mins, slopes = [], [], []

for _, trial in df_trials.iterrows():
    stim_time = trial['stimOnTrigger_times']
    start, end = stim_time + pre_window[0], stim_time + pre_window[1]
    mask = (df_nph['times'] >= start) & (df_nph['times'] < end)
    segment = df_nph.loc[mask, 'zdFF']

    if segment.empty:
        means.append(np.nan)
        mins.append(np.nan)
        slopes.append(np.nan)
    else:
        means.append(segment.mean())
        mins.append(segment.min())
        # slope = linear regression on time vs zdFF
        t = df_nph.loc[mask, 'times'].values.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(t, segment.values)
        slopes.append(lr.coef_[0])

df_trials['zdff_drop_pre_stim_mean'] = means
df_trials['zdff_drop_pre_stim_min'] = mins
df_trials['zdff_drop_pre_stim_slope'] = slopes


# ==========================================================
# 2. FEATURE ENGINEERING
# ==========================================================

# ---- Current trial (A)
df_trials['quiescencePeriod'] = df_trials['quiescencePeriod']
df_trials['quiescenceTime']   = df_trials['quiescenceTime']
df_trials['trialTime']        = df_trials['trialTime']

# ---- Previous trial (B)
for col in ['quiescenceTime', 'trialTime', 'feedbackType', 'choice',
            'allContrasts', 'allSContrasts', 'reactionTime',
            'probabilityLeft', 'responseTime',
            'zdff_drop_pre_stim_mean', 'zdff_drop_pre_stim_min', 'zdff_drop_pre_stim_slope']:
    df_trials[f'prev_{col}'] = df_trials[col].shift(1)

# ---- Sequential (C)
df_trials['correct'] = (df_trials['feedbackType'] == 1).astype(int)
# Consecutive correct
consec_correct, consec_incorrect = [], []
count_c, count_i = 0, 0
for fb in df_trials['feedbackType']:
    consec_correct.append(count_c)
    consec_incorrect.append(count_i)
    if fb == 1:
        count_c += 1
        count_i = 0
    elif fb == -1:
        count_i += 1
        count_c = 0
    else:
        count_c = count_i = 0

df_trials['consecutive_correct'] = consec_correct
df_trials['consecutive_incorrect'] = consec_incorrect

# ---- Interactions (D)
df_trials['interaction_quiescence_correct'] = df_trials['quiescenceTime'] * df_trials['correct']
df_trials['interaction_reaction_correct'] = df_trials['reactionTime'] * df_trials['correct']
df_trials['interaction_prev_trialTime_quiescenceTime'] = df_trials['prev_trialTime'] * df_trials['quiescenceTime']
df_trials['interaction_prev_feedbackType_quiescenceTime'] = df_trials['prev_feedbackType'] * df_trials['quiescenceTime']



# ==========================================================
# 3. SELECT FINAL FEATURE SET
# ==========================================================

feature_cols = [
    # Current trial
    'quiescencePeriod', 'quiescenceTime', 'trialTime',
    # Previous trial
    'prev_quiescenceTime', 'prev_trialTime', 'prev_feedbackType',
    'prev_choice', 'prev_allContrasts', 'prev_allSContrasts',
    'prev_reactionTime', 'prev_probabilityLeft', 'prev_responseTime',
    'prev_zdff_drop_pre_stim_mean', 'prev_zdff_drop_pre_stim_min', 'prev_zdff_drop_pre_stim_slope',
    # Sequential
    'consecutive_correct', 'consecutive_incorrect',
    # Interactions
    'interaction_quiescence_correct', 'interaction_reaction_correct',
    'interaction_prev_trialTime_quiescenceTime', 'interaction_prev_feedbackType_quiescenceTime'
]

# # fewer features 
# feature_cols = [
#     # Current trial
#     'quiescenceTime',
#     # Previous trial
#     'prev_feedbackType',
#     'prev_choice', 'prev_allSContrasts',
#     'prev_zdff_drop_pre_stim_mean', 'prev_zdff_drop_pre_stim_min', 'prev_zdff_drop_pre_stim_slope',
#     # Sequential
#     'consecutive_correct', 'consecutive_incorrect',
#     # Interactions
#     # 'interaction_quiescence_correct', 'interaction_reaction_correct',
#     # 'interaction_prev_trialTime_quiescenceTime', 'interaction_prev_feedbackType_quiescenceTime'
# ]

# Keep only numeric features
X = df_trials[feature_cols].select_dtypes(include=[np.number]).fillna(0)

targets = {
    "Mean": df_trials['zdff_drop_pre_stim_mean'],
    "Min": df_trials['zdff_drop_pre_stim_min'],
    "Slope": df_trials['zdff_drop_pre_stim_slope'],
}

# ==========================================================
# 4. TRAINING LOOP FOR MULTIPLE TARGETS
# ==========================================================

models = {
    "LinearRegression": LinearRegression(),
    "LassoCV": LassoCV(cv=5, random_state=42),
    "ElasticNetCV": ElasticNetCV(cv=5, random_state=42),
    "SVR": SVR(kernel='rbf'),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, random_state=42),
    "NeuralNet": MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu', solver='adam',
        learning_rate_init=0.001,
        max_iter=500, random_state=42
    )
}

all_results = {}

for target_name, y in targets.items():
    print(f"\n==================== Target: {target_name} ====================")

    mask = y.notna()
    X_target, y_target = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_target, y_target, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for name, model in models.items():
        if name in ["LinearRegression", "LassoCV", "ElasticNetCV", "SVR"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        print(f"=== {name} ===")
        print(f"R²: {r2:.3f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}\n")

        # Feature importance (tree models only)
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=X.columns)
            importances = importances.sort_values(ascending=False)
            plt.figure(figsize=(8, 6))
            sns.barplot(x=importances, y=importances.index, color="skyblue")
            plt.title(f"Feature Importances ({name}) - {target_name}")
            plt.tight_layout()
            plt.show()

            # SHAP for tree models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values, X_train, feature_names=X.columns, show=False)
            plt.title(f"SHAP Summary - {name} ({target_name})")
            plt.show()

            # PDP for top 3 features
            top_features = importances.index[:3].tolist()
            fig, ax = plt.subplots(figsize=(12, 6))
            PartialDependenceDisplay.from_estimator(
                model, X_train, top_features, feature_names=X.columns, ax=ax
            )
            plt.suptitle(f"PDP ({name}) - {target_name}", fontsize=14)
            plt.show()

    all_results[target_name] = results

# Final results summary
print("\n===== ALL RESULTS =====")
for target, res in all_results.items():
    print(f"\nTarget: {target}")
    for model, metrics in res.items():
        print(f"{model}: R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

# %%
# ==========================================================
# 5. Select Best Model per Target
# ==========================================================

print("\n===== BEST MODELS PER TARGET =====")
best_models = {}

for target, res in all_results.items():
    # Pick model with highest R²
    best_model = max(res.items(), key=lambda x: x[1]['R2'])
    best_models[target] = best_model
    print(f"{target}: {best_model[0]} (R²={best_model[1]['R2']:.3f}, "
          f"RMSE={best_model[1]['RMSE']:.4f}, MAE={best_model[1]['MAE']:.4f})")
    


# %% 
""" 
Feature Importance - double-check! 
"""

feature_cols
X = df_trials[feature_cols].select_dtypes(include=[np.number]).fillna(0)
X = X.fillna(0)  # fill NaNs (first trial, etc.)
y = df_trials['zdff_drop_pre_stim_mean']


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
#KB TO EXPLORE
alpha, gamma = 0.2, 0.9  # learning rate, discount factor
q_values, reward_preds = [], []
Q = {0:0, 1:0}  # left=0, right=1

for _, row in df_trials.iterrows():
    choice_raw = row['choice']
    reward = row['feedbackType']
    
    # Map choices: -1 → 0 (left), +1 → 1 (right)
    if choice_raw == -1:
        choice = 0
    elif choice_raw == 1:
        choice = 1
    else:
        # If NaN or invalid choice, skip with default value
        q_values.append(np.nan)
        reward_preds.append(np.nan)
        continue
    
    # Append current values
    q_values.append(Q[choice])
    reward_preds.append(Q[1] - Q[0])  # relative bias
    
    # Q-learning update
    pe = reward - Q[choice]
    Q[choice] += alpha * pe

df_trials['q_value'] = q_values
df_trials['q_bias'] = reward_preds
plt.plot(df_trials.q_bias)
plt.plot(df_trials.probabilityLeft) 










# %% 
""" 
ADDING UNSUPERVISED MODELS 
21-August-2025
"""

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Choose features for clustering
feature_cols = [
    # Current trial
    'quiescenceTime',
    'prev_trialTime', 'prev_feedbackType',
    'prev_choice', 'prev_allSContrasts',
    'prev_reactionTime', 'prev_probabilityLeft', 
    # Sequential
    'consecutive_correct', 'consecutive_incorrect',
    # Interactions
    'interaction_prev_trialTime_quiescenceTime', 'interaction_prev_feedbackType_quiescenceTime'
]
cluster_features = df_trials[feature_cols].dropna()

# Scale
scaler = StandardScaler()
X_cluster = scaler.fit_transform(cluster_features)



from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Candidate k values
K = range(2, 11)
sil_scores, ch_scores, db_scores, inertias = [], [], [], []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_cluster)
    labels = kmeans.labels_
    
    sil_scores.append(silhouette_score(X_cluster, labels))
    ch_scores.append(calinski_harabasz_score(X_cluster, labels))
    db_scores.append(davies_bouldin_score(X_cluster, labels))
    inertias.append(kmeans.inertia_)

# --- Plot results ---
fig, axs = plt.subplots(2, 2, figsize=(10,8))

axs[0,0].plot(K, sil_scores, 'o-')
axs[0,0].set_title("Silhouette Score (higher=better)")
axs[0,0].set_xlabel("k")

axs[0,1].plot(K, ch_scores, 'o-')
axs[0,1].set_title("Calinski-Harabasz Index (higher=better)")
axs[0,1].set_xlabel("k")

axs[1,0].plot(K, db_scores, 'o-')
axs[1,0].set_title("Davies-Bouldin Index (lower=better)")
axs[1,0].set_xlabel("k")

axs[1,1].plot(K, inertias, 'o-')
axs[1,1].set_title("Elbow method (Inertia, lower=better)")
axs[1,1].set_xlabel("k")

plt.tight_layout()
plt.show()

"""
Silhouette: pick the peak.

Calinski–Harabasz: also peak.

Davies–Bouldin: look for the dip.

Inertia: look for the “elbow” where extra clusters stop helping much.
"""


""" BEST CLUSTER NUMBER = 4 """

# Try k-means
kmeans = KMeans(n_clusters=4, random_state=42)
df_trials.loc[cluster_features.index, "cluster_kmeans"] = kmeans.fit_predict(X_cluster)

# Try Gaussian Mixture
gmm = GaussianMixture(n_components=4, random_state=42)
df_trials.loc[cluster_features.index, "cluster_gmm"] = gmm.fit_predict(X_cluster)

# Quick visualization (PCA)
X_pca = PCA(n_components=2).fit_transform(X_cluster)
plt.figure(figsize=(6,5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df_trials.loc[cluster_features.index,"cluster_kmeans"],
                palette="viridis")
plt.title("Clustering by quiescence/zdFF features")
plt.show()

# Compare zdFF across clusters
sns.boxplot(x="cluster_kmeans", y="zdff_drop_pre_stim_mean", data=df_trials)
plt.title("Pre-stim zdFF mean by cluster")
plt.show()


sns.boxplot(x="cluster_kmeans", y="quiescenceTime", data=df_trials)
plt.title("Quiescence duration by cluster")
plt.show()

sns.countplot(x="cluster_kmeans", hue="prev_feedbackType", data=df_trials)
plt.title("Correct vs Incorrect trials per cluster")
plt.show()

sns.countplot(x="cluster_kmeans", hue="prev_choice", data=df_trials)
plt.title("Choice distribution per cluster")
plt.show()




#%%
import numpy as np
# Step 1: shift so that value at t=0 = 0
# (assuming zdff_drop_pre_stim_mean is already computed relative to t=0 baseline,
# but if not, you can subtract the value at t=0)
df["zdff_drop_norm_mean"] = df["zdff_drop_pre_stim_mean"] - df["zdff_drop_pre_stim_mean"].iloc[0]
df["zdff_drop_norm_min"] = df["zdff_drop_pre_stim_min"] - df["zdff_drop_pre_stim_min"].iloc[0]
df["zdff_drop_norm_slope"] = df["zdff_drop_pre_stim_slope"] - df["zdff_drop_pre_stim_slope"].iloc[0]

# Step 2: shift upwards so all values ≥ 0
df["zdff_drop_norm_mean"] = df["zdff_drop_norm_mean"] - df["zdff_drop_norm_mean"].min()

# Do the same for min and slope
df["zdff_drop_norm_min"] = df["zdff_drop_pre_stim_min"] - df["zdff_drop_pre_stim_min"].min()
df["zdff_drop_norm_slope"] = df["zdff_drop_pre_stim_slope"] - df["zdff_drop_pre_stim_slope"].min()



#%%














































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
