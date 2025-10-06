#%%
"""
KB
16_V3_redoing
06-October-2025

trying to solve the TTL mismatch seen in the psth plots of some (task_00) sessions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from brainbox.behavior.training import compute_performance 
from brainbox.io.one import SessionLoader 
import ibldsp.utils
import scipy.signal
from iblutil.numerical import rcoeff
from functions import *
import sys
sys.path.insert(0, "/home/kceniabougrova/Documents/GitHub/ibl-photometry/src")

from one.api import ONE 
one = ONE() 

table_path = '/home/kceniabougrova/Downloads/new_sessions_Aug2025 - all_together.csv'
sessions_list = pd.read_csv(table_path)


i = 1000 #select the session here #DA
# i = 1626


eid, subject, date, region, nph_file_path, nph_bnc_path = select_one_session(
    sessions_list, row=i)


old_prefix = '/mnt/h0/kb/'
new_prefix = '/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/'

nph_file_path = nph_file_path.replace(old_prefix, new_prefix)
nph_bnc_path = nph_bnc_path.replace(old_prefix, new_prefix)

# Load the CSVs
df_nph = pd.read_csv(nph_file_path)
df_bnc = pd.read_csv(nph_bnc_path)

# Load the behavior
# df_trials, subject, session_date = load_trials_updated(eid)
# df_trials, subject, session_date = load_trials_updated_for_sessions_with_task00(eid)

try:
    df_trials, subject, session_date = load_trials_updated(eid)
    if df_trials is None or df_trials.empty:
        raise ValueError("df_trials is empty")
    print("✅ df_trials created successfully with load_trials_updated")
except Exception as e:
    print(f"⚠️ load_trials_updated failed: {e}")
    print("👉 Trying load_trials_updated_for_sessions_with_task00...")
    df_trials, subject, session_date = load_trials_updated_for_sessions_with_task00(eid)

print(f"Final df_trials shape: {df_trials.shape if df_trials is not None else None}")
print(subject, date, region)

# %%
df_nph["mouse"] = subject
df_nph["date"] = session_date
df_nph["region"] = region
df_nph["eid"] = eid 
#%%
# Remove 'Value.' prefix from columns
df_bnc.columns = [col.replace("Value.", "") for col in df_bnc.columns]
df_bnc = df_bnc[df_bnc["Value"] == True].reset_index(drop=True)

#%% 
tph = df_bnc["Timestamp"].values
tbpod = df_trials['stimOnTrigger_times'].values
# tbpod = df_trials['stimOnTrigger_times'].dropna().values

# ---
ratio = len(tph) / len(tbpod)
print(f"TTL ratio tph/tbpod = {ratio:.2f}")

if np.isclose(ratio, 1.0, rtol=0.1):   # ~equal length (within ±10%)
    print("✅ 1:1 sync — using stimOnTrigger_times only")
    fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)

elif ratio > 1.5:
    print("⚠️ Multiple photometry TTLs per trial — using intervals + feedback for sync")
    tbpod = np.sort(np.r_[
        df_trials['intervals_0'].dropna().values,
        df_trials['intervals_1'].dropna().values - 1,
        df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].dropna().values
    ])
    tph = tph[15:]  # skip early noise TTLs
    fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)

else:
    raise ValueError(f"❌ Unexpected mismatch: tph={len(tph)}, tbpod={len(tbpod)}, ratio={ratio:.2f}")

assert abs(drift_ppm) < 100, f"Drift too high: {drift_ppm:.3f} ppm"

# Apply mapping
df_nph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_nph["Timestamp"])

plt.plot(df_nph["Timestamp"], df_nph["bpod_frame_times"])
plt.xlabel("Photometry timestamps")
plt.ylabel("Bpod mapped time")
plt.title(f"Drift = {drift_ppm:.3f} ppm")
plt.show()


# %%
session_start = df_trials.intervals_0.values[0] - 10  # Start time, 100 seconds before the first tph value
session_end = df_trials.intervals_1.values[-1] + 10   # End time, 100 seconds after the last tph value

# Select data within the specified time range
selected_data = df_nph[
    (df_nph['bpod_frame_times'] >= session_start) &
    (df_nph['bpod_frame_times'] <= session_end)
] 
df_nph = selected_data.reset_index(drop=True) 




print("Len TTL: ", len(tph), "Len tbpod: ", len(tbpod), "Len trials: ", len(df_trials))




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


# %%
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
raw_TTL_bpod = tbpod
raw_TTL_nph = tph

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
# %%
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
# %%
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
# %%
zdFF = (z_signal - z_reference_fitted)



fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(zdFF,'black')


df_nph['zdFF'] = zdFF
nph = df_nph



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
# %%
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
# %%
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
# %%
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
# %%
# Plot trials separately
plt.plot(time_axis, photometry_feedback[:, mask_correct], color='blue', linewidth=0.3, alpha=0.3)
plt.plot(time_axis, photometry_feedback[:, mask_incorrect], color='red', linewidth=0.3, alpha=0.3)

# Event marker
plt.axvline(x=0, color='black', linestyle='--')

plt.xlabel("Time (s)")
plt.ylabel("ΔF/F (z-scored)")
plt.title("Peri-feedback PSTH split by feedback type")
plt.ylim(-2,5)
plt.show()
# %%
# Mask for correct vs incorrect/other
mask_correct = df_trials.feedbackType.values == 1
mask_incorrect = ~mask_correct

# Plot trials separately
plt.plot(time_axis, photometry_feedback[:, mask_correct], color='blue', linewidth=0.3, alpha=0.2)
plt.plot(time_axis, photometry_feedback[:, mask_incorrect], color='red', linewidth=0.3, alpha=0.2) 

# --- Mean traces on top ---
mean_correct = np.nanmean(photometry_feedback[:, mask_correct], axis=1)
mean_incorrect = np.nanmean(photometry_feedback[:, mask_incorrect], axis=1)

plt.plot(time_axis, mean_correct, color='blue', linewidth=2, label="Correct (mean)")
plt.plot(time_axis, mean_incorrect, color='red', linewidth=2, label="Incorrect (mean)")

# Event marker
plt.axvline(x=0, color='black', linestyle='--')

plt.xlabel("Time (s)")
plt.ylabel("ΔF/F (z-scored)")
plt.title("Peri-feedback PSTH split by feedback type")
plt.ylim(-2.5,3)
plt.show()
# %%
# Mask for correct vs incorrect/other
mask_correct = df_trials.feedbackType.values == 1
mask_incorrect = ~mask_correct

# Plot trials separately
plt.plot(time_axis, photometry_feedback[:, mask_correct], color='blue', linewidth=0.3, alpha=0.2)
plt.plot(time_axis, photometry_feedback[:, mask_incorrect], color='red', linewidth=0.3, alpha=0.2) 

# --- Mean traces on top ---
mean_correct = np.nanmean(photometry_feedback[:, mask_correct], axis=1)
mean_incorrect = np.nanmean(photometry_feedback[:, mask_incorrect], axis=1)

plt.plot(time_axis, mean_correct, color='blue', linewidth=3, label="Correct (mean)")
plt.plot(time_axis, mean_incorrect, color='red', linewidth=3, label="Incorrect (mean)")

# Event marker
plt.axvline(x=0, color='black', linestyle='--')

plt.xlabel("Time (s)")
plt.ylabel("ΔF/F (z-scored)")
plt.title("Peri-feedback PSTH split by feedback type")
plt.ylim(-2.5,5)
plt.show()
# %%
