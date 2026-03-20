#%%
"""
KB
16_V3_redoing
06-October-2025

 Solved the TTL mismatch seen in the psth plots of some (task_00) sessions

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
# from brainbox.behavior.training import compute_performance 
# from brainbox.io.one import SessionLoader 
# import ibldsp.utils
# import scipy.signal
# from iblutil.numerical import rcoeff
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




# Filter the sessions by probabilityLeft and count the trial number 
allowed_sets = [
    {0.5, 0.2},
    {0.5, 0.8},
    {0.5, 0.2, 0.8}
]

filtered_sessions = []

for idx, row in sessions_list.iterrows():
    try:
        eid, subject, date, region, nph_file_path, nph_bnc_path = select_one_session(sessions_list, row=idx)
        nph_file_path = nph_file_path.replace(old_prefix, new_prefix)
        nph_bnc_path = nph_bnc_path.replace(old_prefix, new_prefix)

        try:
            df_trials, subject, session_date = load_trials_updated(eid)
            if df_trials is None or df_trials.empty:
                raise ValueError("empty df_trials")
        except Exception:
            df_trials, subject, session_date = load_trials_updated_for_sessions_with_task00(eid)

        probs = set(np.round(df_trials["probabilityLeft"].dropna().unique(), 2))
        if probs in allowed_sets:
            row_dict = row.to_dict()
            row_dict["n_trials"] = len(df_trials)
            filtered_sessions.append(row_dict)
            print(f"✅ Session {idx} ({subject}, {date}) added: probs={probs}")
        else:
            print(f"❌ Session {idx} skipped: probs={probs}")

    except Exception as e:
        print(f"⚠️ Failed on session {idx}: {e}")

filtered_sessions_df = pd.DataFrame(filtered_sessions)
print(f"\n✅ Total filtered sessions: {len(filtered_sessions_df)}")

# Optional: save it for later
# filtered_sessions_df.to_csv("/home/kceniabougrova/Downloads/filtered_sessions.csv", index=False)


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





















































































#%%
#%%
"""
KB
16_V3_redoing
06-October-2025

 Solved the TTL mismatch seen in the psth plots of some (task_00) sessions

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
# from brainbox.behavior.training import compute_performance 
# from brainbox.io.one import SessionLoader 
# import ibldsp.utils
# import scipy.signal
# from iblutil.numerical import rcoeff
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



#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIG
# =========================================================
PLOT = True
old_prefix = '/mnt/h0/kb/'
new_prefix = '/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/'

processed_sessions = []
error_sessions_processing = []

# =========================================================
# LOOP THROUGH FILTERED SESSIONS
# =========================================================
start_idx = 1851 #1566 #1209 #1159 #1138 #1089 #1073 #1058 #987 #921 #608 #523 #345 #149

for idx, row in sessions_list.iloc[start_idx:].iterrows():
    try:
        print(f"\n====================")
        print(f"Processing session {idx}: {row['subject']} {row['date']} {row['region']}")
        print("====================")

        # =========================================================
        # Retrieve all session info (adds nph paths automatically)
        # =========================================================
        eid, subject, date, region, nph_file_path, nph_bnc_path = select_one_session(
            sessions_list, row=idx
        )

        # Fix paths
        nph_file_path = nph_file_path.replace(old_prefix, new_prefix)
        nph_bnc_path = nph_bnc_path.replace(old_prefix, new_prefix)

        # =========================================================
        # Load data
        # =========================================================
        df_nph = pd.read_csv(nph_file_path)
        df_bnc = pd.read_csv(nph_bnc_path)

        # Load behavior (try both)
        try:
            df_trials, subject, session_date = load_trials_updated(eid)
            if df_trials is None or df_trials.empty:
                raise ValueError("empty df_trials")
        except Exception:
            df_trials, subject, session_date = load_trials_updated_for_sessions_with_task00(eid)

        # =========================================================
        # Attach metadata
        # =========================================================
        df_nph["mouse"] = subject
        df_nph["date"] = session_date
        df_nph["region"] = region
        df_nph["eid"] = eid

        # --- Clean BNC
        df_bnc.columns = [col.replace("Value.", "") for col in df_bnc.columns]
        df_bnc = df_bnc[df_bnc["Value"] == True].reset_index(drop=True)

        tph = df_bnc["Timestamp"].values
        tbpod = df_trials['stimOnTrigger_times'].values

        ratio = len(tph) / len(tbpod)
        print(f"TTL ratio tph/tbpod = {ratio:.2f}")

        if np.isclose(ratio, 1.0, rtol=0.1):
            fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)
        elif ratio > 1.5:
            tbpod = np.sort(np.r_[
                df_trials['intervals_0'].dropna().values,
                df_trials['intervals_1'].dropna().values - 1,
                df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].dropna().values
            ])
            tph = tph[15:]
            fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)
        else:
            raise ValueError(f"Unexpected mismatch: tph={len(tph)}, tbpod={len(tbpod)}, ratio={ratio:.2f}")

        assert abs(drift_ppm) < 100, f"Drift too high: {drift_ppm:.3f} ppm"

        # Apply mapping
        df_nph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_nph["Timestamp"])

        # --- Crop by trial window
        session_start = df_trials.intervals_0.values[0] - 10
        session_end = df_trials.intervals_1.values[-1] + 10
        df_nph = df_nph[
            (df_nph['bpod_frame_times'] >= session_start) &
            (df_nph['bpod_frame_times'] <= session_end)
        ].reset_index(drop=True)

        print(f"✅ Sync and crop done for {subject} {date}")

        # =========================================================
        # LED + Signal preprocessing
        # =========================================================
        df_nph = LedState_or_Flags(df_nph)
        df_470, df_415 = verify_length(df_nph)
        verify_repetitions(df_nph["LedState"])

        colname = "LedState" if "LedState" in df_nph.columns else "Flags"

        # clean edges
        if df_nph[colname].iloc[0] == 1 and df_nph[colname].iloc[-1] == 1:
            df_nph = df_nph.iloc[1:]
        if df_nph[colname].iloc[0] == 2 and df_nph[colname].iloc[-1] == 2:
            df_nph = df_nph.iloc[:-2]

        df_470 = df_nph[df_nph[colname] == 2].reset_index(drop=True)
        df_415 = df_nph[df_nph[colname] == 1].reset_index(drop=True)
        assert len(df_470) == len(df_415), "GCaMP and isosbestic signals length mismatch"

        time_diffs = df_470["Timestamp"].diff().dropna()
        fs = 1 / time_diffs.median()

        raw_reference = df_415[region].values
        raw_signal = df_470[region].values
        raw_timestamps_bpod = df_470["bpod_frame_times"]

        my_array = np.column_stack((raw_timestamps_bpod, raw_reference, raw_signal))

        df_nph = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium']) #IMPORTANT DF


        raw_reference = df_nph['raw_isosbestic'][0:]
        raw_signal = df_nph['raw_calcium'][0:]

        smooth_win = 10
        smooth_reference = smooth_signal(raw_reference, smooth_win)
        smooth_calcium = smooth_signal(raw_signal, smooth_win)



        lambd = 5e4 # Adjust lambda to get the best fit
        porder = 1
        itermax = 50
        r_base=airPLS(smooth_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
        s_base=airPLS(smooth_calcium,lambda_=lambd,porder=porder,itermax=itermax)



        remove = 0
        ref_corrected = smooth_reference[remove:] - r_base[remove:]
        sig_corrected = smooth_calcium[remove:] - s_base[remove:]

        z_reference = (ref_corrected - np.median(ref_corrected)) / np.std(ref_corrected)
        z_signal = (sig_corrected - np.median(sig_corrected)) / np.std(sig_corrected)




        from sklearn.linear_model import Lasso
        lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                    positive=True, random_state=9999, selection='random')
        n = len(z_reference)
        lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))

        z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)


        zdFF = (z_signal - z_reference_fitted)



        df_nph['zdFF'] = zdFF
        nph = df_nph


        # =========================================================
        #  PSTH plot (Peri-feedback activity)
        # =========================================================

        if PLOT:
            try:
                from scipy.stats import zscore

                # --- Basic parameters
                EVENT = "feedback_times"          # can be "stimOnTrigger_times" or "feedback_times"
                time_bef, time_aft = -1, 2        # peri-event window
                PERIEVENT_WINDOW = [time_bef, time_aft]
                SAMPLING_RATE = int(1 / np.mean(np.diff(df_nph.times)))

                # --- Align zdFF to events
                t = df_nph["times"].values
                calcium = df_nph["zdFF"].values
                t_events = df_trials[EVENT].dropna().values

                # Create peri-event index matrix
                n_trials = len(t_events)
                samples_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE,
                                        PERIEVENT_WINDOW[1] * SAMPLING_RATE)
                psth_idx = np.tile(samples_window[:, np.newaxis], (1, n_trials))
                event_idx = np.searchsorted(t, t_events)
                psth_idx += event_idx

                # Mask invalid indices
                psth_idx = psth_idx[(psth_idx >= 0) & (psth_idx < len(t))].reshape(-1, n_trials)

                # Compute PSTH (peri-event matrix)
                photometry_feedback = calcium[psth_idx]

                # Build time axis
                n_timepoints = photometry_feedback.shape[0]
                time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

                # --- Plot trial-wise and mean traces
                plt.figure(figsize=(8, 6))
                mask_correct = df_trials.feedbackType.values == 1
                mask_incorrect = ~mask_correct

                # Plot all trials (optional)
                plt.plot(time_axis, photometry_feedback[:, mask_correct], color='blue', alpha=0.1, linewidth=0.5)
                plt.plot(time_axis, photometry_feedback[:, mask_incorrect], color='red', alpha=0.1, linewidth=0.5)

                # Mean ± SEM
                mean_correct = np.nanmean(photometry_feedback[:, mask_correct], axis=1)
                mean_incorrect = np.nanmean(photometry_feedback[:, mask_incorrect], axis=1)
                sem_correct = np.nanstd(photometry_feedback[:, mask_correct], axis=1) / np.sqrt(mask_correct.sum())
                sem_incorrect = np.nanstd(photometry_feedback[:, mask_incorrect], axis=1) / np.sqrt(mask_incorrect.sum())

                plt.plot(time_axis, mean_correct, color='blue', linewidth=2, label='Correct')
                plt.fill_between(time_axis, mean_correct - sem_correct, mean_correct + sem_correct,
                                color='blue', alpha=0.3)
                plt.plot(time_axis, mean_incorrect, color='red', linewidth=2, label='Incorrect')
                plt.fill_between(time_axis, mean_incorrect - sem_incorrect, mean_incorrect + sem_incorrect,
                                color='red', alpha=0.3)

                plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
                plt.title(f"{idx} - PSTH peri-{EVENT} — {subject} {region} {date}")
                plt.xlabel("Time (s)")
                plt.ylabel("ΔF/F (z-scored)")
                plt.legend(frameon=False)
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)  # ensures plot shows before next loop iteration

            except Exception as e:
                print(f"⚠️ Could not plot PSTH for {subject} {date} {region}: {e}")








        processed_sessions.append({
            "subject": subject,
            "date": date,
            "region": region,
            "eid": eid,
            "fs": fs,
            "n_frames": len(df_nph)
        })



    except Exception as e:
        print(f"⚠️ Error during processing of session {idx}: {e}")
        err = row.to_dict()
        err["error"] = str(e)
        error_sessions_processing.append(err)

# =========================================================
# Wrap up
# =========================================================
processed_sessions_df = pd.DataFrame(processed_sessions)
df_errors_processing = pd.DataFrame(error_sessions_processing)

print(f"\n✅ Completed processing {len(processed_sessions_df)} sessions")
print(f"⚠️ {len(df_errors_processing)} sessions had errors")










# %%
#%%
"""
KB
16_V3_redoing
07-October-2025

 Solved the TTL mismatch seen in the psth plots of some (task_00) sessions
 Filter for all the G sessions 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
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


# =========================================================
# CONFIG
# =========================================================
PLOT = True
old_prefix = '/mnt/h0/kb/'
new_prefix = '/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/'

processed_sessions = []
error_sessions_processing = []


# =========================================================
# LOOP THROUGH FILTERED SESSIONS
# =========================================================
start_idx = 0 #1566 #1209 #1159 #1138 #1089 #1073 #1058 #987 #921 #608 #523 #345 #149


# =========================================================
# LOAD AND FILTER GOOD SESSIONS
# =========================================================
excel_path = '/home/kceniabougrova/Downloads/new_sessions_Aug2025 - all_together_tab2.xlsx'
df_all = pd.read_excel(excel_path, sheet_name='new_sessions_Aug2025 - all_to-1')
df_good = df_all[df_all["good"] == "G"].reset_index(drop=True)
print(f"✅ Loaded {len(df_good)} good sessions")

# SAVE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs"
# os.makedirs(SAVE_DIR, exist_ok=True)


for idx, row in df_good.iloc[start_idx:].iterrows():
    try:
        print(f"\n====================")
        print(f"Processing session {idx}: {row['subject']} {row['date']} {row['region']}")
        print("====================")

        # =========================================================
        # Retrieve all session info (adds nph paths automatically)
        # =========================================================
        eid, subject, date, region, nph_file_path, nph_bnc_path = select_one_session(
            df_good, row=idx
        )

        # Fix paths
        nph_file_path = nph_file_path.replace(old_prefix, new_prefix)
        nph_bnc_path = nph_bnc_path.replace(old_prefix, new_prefix)

        # =========================================================
        # Load data
        # =========================================================
        df_nph = pd.read_csv(nph_file_path)
        df_bnc = pd.read_csv(nph_bnc_path)

        # Load behavior (try both)
        try:
            df_trials, subject, session_date = load_trials_updated(eid)
            if df_trials is None or df_trials.empty:
                raise ValueError("empty df_trials")
        except Exception:
            df_trials, subject, session_date = load_trials_updated_for_sessions_with_task00(eid)
        # --- Save df_trials
        save_trials_path = f"/home/kceniabougrova/Downloads/good_sessions_outputs/df_trials_{idx}_{subject}_{date}_{region}_{eid}.csv"
        df_trials.to_csv(save_trials_path, index=False)
        print(f"💾 Saved df_trials -> {save_trials_path}")

        # =========================================================
        # Attach metadata
        # =========================================================
        df_nph["mouse"] = subject
        df_nph["date"] = session_date
        df_nph["region"] = region
        df_nph["eid"] = eid

        # --- Clean BNC
        df_bnc.columns = [col.replace("Value.", "") for col in df_bnc.columns]
        df_bnc = df_bnc[df_bnc["Value"] == True].reset_index(drop=True)

        tph = df_bnc["Timestamp"].values
        tbpod = df_trials['stimOnTrigger_times'].values

        ratio = len(tph) / len(tbpod)
        print(f"TTL ratio tph/tbpod = {ratio:.2f}")

        if np.isclose(ratio, 1.0, rtol=0.1):
            fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)
        elif ratio > 1.5:
            tbpod = np.sort(np.r_[
                df_trials['intervals_0'].dropna().values,
                df_trials['intervals_1'].dropna().values - 1,
                df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].dropna().values
            ])
            tph = tph[15:]
            fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)
        else:
            raise ValueError(f"Unexpected mismatch: tph={len(tph)}, tbpod={len(tbpod)}, ratio={ratio:.2f}")

        assert abs(drift_ppm) < 100, f"Drift too high: {drift_ppm:.3f} ppm"

        # Apply mapping
        df_nph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_nph["Timestamp"])

        # --- Crop by trial window
        session_start = df_trials.intervals_0.values[0] - 10
        session_end = df_trials.intervals_1.values[-1] + 10
        df_nph = df_nph[
            (df_nph['bpod_frame_times'] >= session_start) &
            (df_nph['bpod_frame_times'] <= session_end)
        ].reset_index(drop=True)

        print(f"✅ Sync and crop done for {subject} {date}")

        # =========================================================
        # LED + Signal preprocessing
        # =========================================================
        df_nph = LedState_or_Flags(df_nph)
        df_470, df_415 = verify_length(df_nph)
        verify_repetitions(df_nph["LedState"])

        colname = "LedState" if "LedState" in df_nph.columns else "Flags"

        # clean edges
        if df_nph[colname].iloc[0] == 1 and df_nph[colname].iloc[-1] == 1:
            df_nph = df_nph.iloc[1:]
        if df_nph[colname].iloc[0] == 2 and df_nph[colname].iloc[-1] == 2:
            df_nph = df_nph.iloc[:-2]

        df_470 = df_nph[df_nph[colname] == 2].reset_index(drop=True)
        df_415 = df_nph[df_nph[colname] == 1].reset_index(drop=True)
        assert len(df_470) == len(df_415), "GCaMP and isosbestic signals length mismatch"

        time_diffs = df_470["Timestamp"].diff().dropna()
        fs = 1 / time_diffs.median()

        raw_reference = df_415[region].values
        raw_signal = df_470[region].values
        raw_timestamps_bpod = df_470["bpod_frame_times"]

        my_array = np.column_stack((raw_timestamps_bpod, raw_reference, raw_signal))

        df_nph = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium']) #IMPORTANT DF


        raw_reference = df_nph['raw_isosbestic'][0:]
        raw_signal = df_nph['raw_calcium'][0:]

        smooth_win = 10
        smooth_reference = smooth_signal(raw_reference, smooth_win)
        smooth_calcium = smooth_signal(raw_signal, smooth_win)



        lambd = 5e4 # Adjust lambda to get the best fit
        porder = 1
        itermax = 50
        r_base=airPLS(smooth_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
        s_base=airPLS(smooth_calcium,lambda_=lambd,porder=porder,itermax=itermax)



        remove = 0
        ref_corrected = smooth_reference[remove:] - r_base[remove:]
        sig_corrected = smooth_calcium[remove:] - s_base[remove:]

        z_reference = (ref_corrected - np.median(ref_corrected)) / np.std(ref_corrected)
        z_signal = (sig_corrected - np.median(sig_corrected)) / np.std(sig_corrected)




        from sklearn.linear_model import Lasso
        lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                    positive=True, random_state=9999, selection='random')
        n = len(z_reference)
        lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))

        z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)


        zdFF = (z_signal - z_reference_fitted)



        df_nph['zdFF'] = zdFF
        nph = df_nph



        # --- Save df_nph
        save_nph_path = f"/home/kceniabougrova/Downloads/good_sessions_outputs/df_nph_{idx}_{subject}_{date}_{region}_{eid}.csv"
        df_nph.to_csv(save_nph_path, index=False)
        print(f"💾 Saved df_nph -> {save_nph_path}")


        # =========================================================
        #  PSTH plot (Peri-feedback activity)
        # =========================================================

        if PLOT:
            try:
                from scipy.stats import zscore

                # --- Basic parameters
                EVENT = "feedback_times"          # can be "stimOnTrigger_times" or "feedback_times"
                time_bef, time_aft = -1, 2        # peri-event window
                PERIEVENT_WINDOW = [time_bef, time_aft]
                SAMPLING_RATE = int(1 / np.mean(np.diff(df_nph.times)))

                # --- Align zdFF to events
                t = df_nph["times"].values
                calcium = df_nph["zdFF"].values
                t_events = df_trials[EVENT].dropna().values

                # Create peri-event index matrix
                n_trials = len(t_events)
                samples_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE,
                                        PERIEVENT_WINDOW[1] * SAMPLING_RATE)
                psth_idx = np.tile(samples_window[:, np.newaxis], (1, n_trials))
                event_idx = np.searchsorted(t, t_events)
                psth_idx += event_idx

                # Mask invalid indices
                psth_idx = psth_idx[(psth_idx >= 0) & (psth_idx < len(t))].reshape(-1, n_trials)

                # Compute PSTH (peri-event matrix)
                photometry_feedback = calcium[psth_idx]

                # Build time axis
                n_timepoints = photometry_feedback.shape[0]
                time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

                # --- Plot trial-wise and mean traces
                plt.figure(figsize=(8, 6))
                mask_correct = df_trials.feedbackType.values == 1
                mask_incorrect = ~mask_correct

                # Plot all trials (optional)
                plt.plot(time_axis, photometry_feedback[:, mask_correct], color='blue', alpha=0.1, linewidth=0.5)
                plt.plot(time_axis, photometry_feedback[:, mask_incorrect], color='red', alpha=0.1, linewidth=0.5)

                # Mean ± SEM
                mean_correct = np.nanmean(photometry_feedback[:, mask_correct], axis=1)
                mean_incorrect = np.nanmean(photometry_feedback[:, mask_incorrect], axis=1)
                sem_correct = np.nanstd(photometry_feedback[:, mask_correct], axis=1) / np.sqrt(mask_correct.sum())
                sem_incorrect = np.nanstd(photometry_feedback[:, mask_incorrect], axis=1) / np.sqrt(mask_incorrect.sum())

                plt.plot(time_axis, mean_correct, color='blue', linewidth=2, label='Correct')
                plt.fill_between(time_axis, mean_correct - sem_correct, mean_correct + sem_correct,
                                color='blue', alpha=0.3)
                plt.plot(time_axis, mean_incorrect, color='red', linewidth=2, label='Incorrect')
                plt.fill_between(time_axis, mean_incorrect - sem_incorrect, mean_incorrect + sem_incorrect,
                                color='red', alpha=0.3)

                plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
                plt.title(f"{idx} - PSTH peri-{EVENT} — {subject} {region} {date}")
                plt.xlabel("Time (s)")
                plt.ylabel("ΔF/F (z-scored)")
                plt.legend(frameon=False)
                plt.ylim(-2,3)
                plt.tight_layout() 
                # --- Save plot
                plot_path = f"/home/kceniabougrova/Downloads/good_sessions_outputs/plot_{idx}_{subject}_{date}_{region}_{eid}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"🖼️  Saved plot -> {plot_path}")

                plt.show(block=False)
                plt.pause(0.1)  # ensures plot shows before next loop iteration

            except Exception as e:
                print(f"⚠️ Could not plot PSTH for {subject} {date} {region}: {e}")


        processed_sessions.append({
            "subject": subject,
            "date": date,
            "region": region,
            "eid": eid,
            "fs": fs,
            "n_frames": len(df_nph)
        })



    except Exception as e:
        print(f"⚠️ Error during processing of session {idx}: {e}")
        err = row.to_dict()
        err["error"] = str(e)
        error_sessions_processing.append(err)

# =========================================================
# Wrap up
# =========================================================
processed_sessions_df = pd.DataFrame(processed_sessions)
df_errors_processing = pd.DataFrame(error_sessions_processing)

print(f"\n✅ Completed processing {len(processed_sessions_df)} sessions")
print(f"⚠️ {len(df_errors_processing)} sessions had errors")



""" ALL SESSIONS PER MOUSE IN 1 PLOT  """

# %%
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
SUBJECT = "ZFM-04019"   # 👈 change to any mouse name you want
EVENT = "feedback_times"
PERIEVENT_WINDOW = [-1, 2]

# =========================================================
# LOAD DF_GOOD
# =========================================================
excel_path = "/home/kceniabougrova/Downloads/new_sessions_Aug2025 - all_together_tab2.xlsx"
xls = pd.ExcelFile(excel_path)
# sheet_target = [s for s in xls.sheet_names if "toge" in s or "GOOD" in s][0]
df_all = pd.read_excel(excel_path, sheet_name='new_sessions_Aug2025 - all_to-1')
df_good = df_all[df_all["good"] == "G"].reset_index(drop=True)

# # Filter rows for the chosen subject
df_subj = df_good[df_good["subject"] == SUBJECT].reset_index(drop=True)
print(f"✅ Found {len(df_subj)} good sessions for {SUBJECT}")

# =========================================================
# LOOP THROUGH ALL SESSIONS FOR THIS SUBJECT
# =========================================================

plt.figure(figsize=(12, 8), dpi=150)   # 👈 create a large figure once

all_means_correct, all_means_incorrect = [], []
time_axis_ref = None

all_means_correct, all_means_incorrect = [], []
time_axis_ref = None

for i, row in df_subj.iterrows():
    try:
        date = str(row["date"])[:10]
        region = row["region"]
        eid = row["eid"]

        # Find matching files in folder
        df_trials_file = [
            f for f in os.listdir(BASE_DIR)
            if f.startswith("df_trials_") and SUBJECT in f and date in f and region in f and eid in f
        ]
        df_nph_file = [
            f for f in os.listdir(BASE_DIR)
            if f.startswith("df_nph_") and SUBJECT in f and date in f and region in f and eid in f
        ]

        if not df_trials_file or not df_nph_file:
            print(f"⚠️ Missing files for {SUBJECT} {date} {region}")
            continue

        df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
        df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

        # --- Basic parameters
        SAMPLING_RATE = int(1 / np.mean(np.diff(df_nph.times)))
        t = df_nph["times"].values
        calcium = df_nph["zdFF"].values
        t_events = df_trials[EVENT].dropna().values

        # Build peri-event index
        n_trials = len(t_events)
        samples_window = np.arange(
            PERIEVENT_WINDOW[0] * SAMPLING_RATE,
            PERIEVENT_WINDOW[1] * SAMPLING_RATE
        )
        psth_idx = np.tile(samples_window[:, np.newaxis], (1, n_trials))
        event_idx = np.searchsorted(t, t_events)
        psth_idx += event_idx
        psth_idx = psth_idx[(psth_idx >= 0) & (psth_idx < len(t))].reshape(-1, n_trials)

        photometry_feedback = calcium[psth_idx]

        # Build time axis
        n_timepoints = photometry_feedback.shape[0]
        time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)
        if time_axis_ref is None:
            time_axis_ref = time_axis

        # Mask trials
        mask_correct = df_trials.feedbackType.values == 1
        mask_incorrect = ~mask_correct

        # Compute mean PSTHs for this session
        mean_correct = np.nanmean(photometry_feedback[:, mask_correct], axis=1)
        mean_incorrect = np.nanmean(photometry_feedback[:, mask_incorrect], axis=1)

        # Store for grand averages
        all_means_correct.append(mean_correct)
        all_means_incorrect.append(mean_incorrect)

        # Plot each session's mean (thin lines)
        # --- Gradient color for session index
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        n_sessions = len(df_subj)
        cmap_correct = cm.get_cmap("viridis_r", n_sessions)
        cmap_incorrect = cm.get_cmap("viridis_r", n_sessions)

        # from matplotlib.colors import LinearSegmentedColormap
        # colors_correct = ["#b7e67e", "#48c2b5", "#2c3e91"]   # light green → aqua → dark blue
        # cmap_correct = LinearSegmentedColormap.from_list("correct_map", colors_correct, N=n_sessions)
        # colors_incorrect = ["#f6e58d", "#f79c42", "#e74c3c", "#8e44ad"]  # yellow → orange → red → purple
        # cmap_incorrect = LinearSegmentedColormap.from_list("incorrect_map", colors_incorrect, N=n_sessions)

        # from matplotlib.colors import LinearSegmentedColormap
        # # Light → dark blue for correct
        # colors_correct = ["#b3d9ff", "#1f77b4", "#001933"]
        # cmap_correct = LinearSegmentedColormap.from_list("correct_map", colors_correct, N=n_sessions)
        # # Light → dark red for incorrect
        # colors_incorrect = ["#ffb3b3", "#d62728", "#330000"]
        # cmap_incorrect = LinearSegmentedColormap.from_list("incorrect_map", colors_incorrect, N=n_sessions)



        color_c = cmap_correct(i / n_sessions)
        color_i = cmap_incorrect(i / n_sessions)

        # Plot each session's mean (thin gradient lines)
        plt.plot(time_axis, mean_correct, color=color_c, alpha=0.7, linewidth=2.5)
        plt.plot(time_axis, mean_incorrect, color=color_i, alpha=0.7, linewidth=2.5)


        print(f"✅ Added session {i}: {SUBJECT} {date} {region}")

    except Exception as e:
        print(f"⚠️ Error processing session {i}: {e}")

# =========================================================
# AGGREGATE ACROSS SESSIONS
# =========================================================
from scipy.interpolate import interp1d

if all_means_correct and all_means_incorrect:
    # Find a reference time axis (the shortest one)
    min_len = min(len(x) for x in all_means_correct + all_means_incorrect)
    print(f"🔧 Resampling all PSTHs to {min_len} points")

    # Interpolate all means to the same time base
    ref_time = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], min_len)

    aligned_correct, aligned_incorrect = [], []
    for mc, mi in zip(all_means_correct, all_means_incorrect):
        old_time = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], len(mc))
        f_mc = interp1d(old_time, mc, kind='linear', fill_value='extrapolate')
        f_mi = interp1d(old_time, mi, kind='linear', fill_value='extrapolate')
        aligned_correct.append(f_mc(ref_time))
        aligned_incorrect.append(f_mi(ref_time))

    all_means_correct = np.array(aligned_correct)
    all_means_incorrect = np.array(aligned_incorrect)

    grand_correct = np.nanmean(all_means_correct, axis=0)
    grand_incorrect = np.nanmean(all_means_incorrect, axis=0)
    time_axis_ref = ref_time

    plt.plot(time_axis_ref, grand_correct, color="blue", linewidth=5.5, label="Grand Mean - Correct")
    plt.plot(time_axis_ref, grand_incorrect, color="red", linewidth=5.5, label="Grand Mean - Incorrect")

    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("ΔF/F (z-scored)")
    plt.title(f"PSTH summary — {SUBJECT} ({len(df_subj)} sessions)")
    plt.legend(frameon=False)
    # plt.ylim(-2.5, 4)
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, f"PSTH_summary_{SUBJECT}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    sm_c = cm.ScalarMappable(cmap=cmap_correct, norm=mcolors.Normalize(vmin=0, vmax=n_sessions))
    sm_i = cm.ScalarMappable(cmap=cmap_incorrect, norm=mcolors.Normalize(vmin=0, vmax=n_sessions))

    cbar_c = plt.colorbar(sm_c, ax=plt.gca(), fraction=0.03, pad=0.04)
    cbar_c.set_label("Session index (Correct, viridis)", rotation=270, labelpad=15)

    cbar_i = plt.colorbar(sm_i, ax=plt.gca(), fraction=0.03, pad=0.04)
    cbar_i.set_label("Session index (Incorrect, magma)", rotation=270, labelpad=15)

    plt.show()
    print(f"🖼️ Saved summary plot -> {out_path}")
else:
    print("⚠️ No sessions found for plotting.")





#%%
""" ALL SESSIONS PER MOUSE IN 1 PLOT AND PER FIBER """
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
EVENT = "feedback_times"
PERIEVENT_WINDOW = [-1, 2]

# =========================================================
# LOAD DF_GOOD
# =========================================================
excel_path = "/home/kceniabougrova/Downloads/new_sessions_Aug2025 - all_together_tab2.xlsx"
xls = pd.ExcelFile(excel_path)
df_all = pd.read_excel(excel_path, sheet_name='new_sessions_Aug2025 - all_to-1')
df_good = df_all[df_all["good"] == "G"].reset_index(drop=True)
print(f"✅ Loaded {len(df_good)} 'good' sessions")

# =========================================================
# FIND SUBJECTS WITH MULTIPLE FIBERS
# =========================================================
subjects_with_multiple_fibers = (
    df_good.groupby("subject")["fiber"]
    .nunique()
    .loc[lambda x: x > 1]
    .index.tolist()
)
print(f"🧠 Subjects with >1 fiber: {subjects_with_multiple_fibers}")

# =========================================================
# LOOP THROUGH SUBJECTS (each fiber separately)
# =========================================================
for SUBJECT in subjects_with_multiple_fibers:
    df_subj_all = df_good[df_good["subject"] == SUBJECT].reset_index(drop=True)
    unique_fibers = df_subj_all["fiber"].dropna().unique()
    print(f"\n🐭 {SUBJECT}: {len(unique_fibers)} fibers -> {unique_fibers}")

    for FIBER in unique_fibers:
        df_subj = df_subj_all[df_subj_all["fiber"] == FIBER].reset_index(drop=True)
        print(f"🔹 Processing {SUBJECT} — {FIBER} ({len(df_subj)} sessions)")

        plt.figure(figsize=(12, 8), dpi=150)

        all_means_correct, all_means_incorrect = [], []
        time_axis_ref = None

        for i, row in df_subj.iterrows():
            try:
                date = str(row["date"])[:10]
                region = row["region"]
                eid = row["eid"]

                # --- Find matching files
                df_trials_file = [
                    f for f in os.listdir(BASE_DIR)
                    if f.startswith("df_trials_") and SUBJECT in f and date in f and region in f and eid in f
                ]
                df_nph_file = [
                    f for f in os.listdir(BASE_DIR)
                    if f.startswith("df_nph_") and SUBJECT in f and date in f and region in f and eid in f
                ]

                if not df_trials_file or not df_nph_file:
                    print(f"⚠️ Missing files for {SUBJECT} {date} {region}")
                    continue

                df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
                df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

                # --- Basic parameters
                SAMPLING_RATE = int(1 / np.mean(np.diff(df_nph.times)))
                t = df_nph["times"].values
                calcium = df_nph["zdFF"].values
                t_events = df_trials[EVENT].dropna().values

                # --- Build peri-event index
                n_trials = len(t_events)
                samples_window = np.arange(
                    PERIEVENT_WINDOW[0] * SAMPLING_RATE,
                    PERIEVENT_WINDOW[1] * SAMPLING_RATE
                )
                psth_idx = np.tile(samples_window[:, np.newaxis], (1, n_trials))
                event_idx = np.searchsorted(t, t_events)
                psth_idx += event_idx
                psth_idx = psth_idx[(psth_idx >= 0) & (psth_idx < len(t))].reshape(-1, n_trials)

                photometry_feedback = calcium[psth_idx]

                # --- Time axis
                n_timepoints = photometry_feedback.shape[0]
                time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)
                if time_axis_ref is None:
                    time_axis_ref = time_axis

                # --- Mask trials
                mask_correct = df_trials.feedbackType.values == 1
                mask_incorrect = ~mask_correct

                mean_correct = np.nanmean(photometry_feedback[:, mask_correct], axis=1)
                mean_incorrect = np.nanmean(photometry_feedback[:, mask_incorrect], axis=1)

                all_means_correct.append(mean_correct)
                all_means_incorrect.append(mean_incorrect)

                # --- Gradient colors
                n_sessions = len(df_subj)
                cmap_correct = plt.cm.get_cmap("viridis_r", n_sessions)
                cmap_incorrect = plt.cm.get_cmap("viridis_r", n_sessions)
                color_c = cmap_correct(i / n_sessions)
                color_i = cmap_incorrect(i / n_sessions)

                # --- Plot session traces
                plt.plot(time_axis, mean_correct, color=color_c, alpha=0.6, linewidth=1.8)
                plt.plot(time_axis, mean_incorrect, color=color_i, alpha=0.6, linewidth=1.8)

                print(f"✅ Added session {i}: {SUBJECT} {date} {region}")

            except Exception as e:
                print(f"⚠️ Error processing {SUBJECT} {FIBER} session {i}: {e}")

        # =========================================================
        # AGGREGATE ACROSS SESSIONS (GRAND MEAN)
        # =========================================================
        if all_means_correct and all_means_incorrect:
            min_len = min(len(x) for x in all_means_correct + all_means_incorrect)
            ref_time = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], min_len)

            aligned_correct, aligned_incorrect = [], []
            for mc, mi in zip(all_means_correct, all_means_incorrect):
                old_time = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], len(mc))
                f_mc = interp1d(old_time, mc, kind='linear', fill_value='extrapolate')
                f_mi = interp1d(old_time, mi, kind='linear', fill_value='extrapolate')
                aligned_correct.append(f_mc(ref_time))
                aligned_incorrect.append(f_mi(ref_time))

            all_means_correct = np.array(aligned_correct)
            all_means_incorrect = np.array(aligned_incorrect)

            grand_correct = np.nanmean(all_means_correct, axis=0)
            grand_incorrect = np.nanmean(all_means_incorrect, axis=0)

            plt.plot(ref_time, grand_correct, color="blue", linewidth=4, label="Grand Mean - Correct")
            plt.plot(ref_time, grand_incorrect, color="red", linewidth=4, label="Grand Mean - Incorrect")

            plt.axvline(0, color="black", linestyle="--", linewidth=1)
            plt.xlabel("Time (s)")
            plt.ylabel("ΔF/F (z-scored)")
            plt.title(f"{SUBJECT} — {FIBER} ({len(df_subj)} sessions)")
            plt.legend(frameon=False)
            plt.tight_layout()

            save_path = os.path.join(BASE_DIR, f"PSTH_{SUBJECT}_{FIBER}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"💾 Saved {save_path}")
        else:
            print(f"⚠️ No valid sessions for {SUBJECT} {FIBER}")














#%%
#%%
#%%
#%%
#%%
""" 
#########################################################################################
KB 08102025
select only the BCW sessions
"""
import os
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
excel_path = "/home/kceniabougrova/Downloads/new_sessions_Aug2025 - all_together_tab2.xlsx"

# =========================================================
# LOAD AND FILTER GOOD SESSIONS
# =========================================================
xls = pd.ExcelFile(excel_path)
df_all = pd.read_excel(excel_path, sheet_name='new_sessions_Aug2025 - all_to-1')

df_good = df_all[df_all["good"] == "G"].reset_index(drop=True)
print(f"✅ Loaded {len(df_good)} 'good' sessions")

# =========================================================
# FIND FILES THAT MATCH THE BCW PROBABILITY CRITERIA
# =========================================================
allowed_sets = [{0.5, 0.2}, {0.5, 0.8}, {0.5, 0.2, 0.8}]
good_sessions_BCW = []

for i, row in df_good.iterrows():
    subject = row["subject"]
    date = str(row["date"])[:10]
    region = row["region"]
    eid = row["eid"]

    # --- Locate corresponding df_trials file
    df_trials_file = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith("df_trials_") and subject in f and date in f and region in f and eid in f
    ]

    if not df_trials_file:
        # No trials file found
        continue

    try:
        df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
        probs = set(np.round(df_trials["probabilityLeft"].dropna().unique(), 2))

        if probs in allowed_sets:
            good_sessions_BCW.append(row)

    except Exception as e:
        print(f"⚠️ Error reading {subject} {date} {region}: {e}")

# =========================================================
# CREATE THE FILTERED DF
# =========================================================
df_good_BCW = pd.DataFrame(good_sessions_BCW).reset_index(drop=True)
print(f"✅ Created df_good_BCW with {len(df_good_BCW)} sessions (biased protocols only)")

# save it
output_path = "/home/kceniabougrova/Downloads/df_good_BCW.xlsx"
df_good_BCW.to_excel(output_path, index=False)
print(f"💾 Saved -> {output_path}")

# =========================================================
# Count rows for each unique pair of subject × fiber
# =========================================================
df_counts = (
    df_good_BCW
    .groupby(['subject', 'fiber'])
    .size()                      # counts rows per group
    .reset_index(name='n_sessions')  # rename the count column
    .sort_values(['subject', 'fiber'])
    .reset_index(drop=True)
)

print("📊 Number of sessions per subject × fiber:")
print(df_counts)

# save it
output_path = "/home/kceniabougrova/Downloads/good_sessions_outputs/df_good_BCW.xlsx"
df_good_BCW.to_excel(output_path, index=False)
print(f"💾 Saved -> {output_path}")













































""" IMPORTANT 10102025 """
#%%
""" 
#########################################################################################
pick only 1 session and plot it 
use predictor modeling 
""" 
df_good_BCW = pd.read_excel("/home/kceniabougrova/Downloads/good_sessions_outputs/df_good_BCW.xlsx")

i = 58  
row = df_good_BCW.iloc[i]

# Extract metadata
subject = row['subject']
date = str(row['date'])[:10]
region = row['region']
fiber = row['fiber']
eid = row['eid']

print(f"📦 Session {i} — {subject} | {date} | {region} | {fiber} | {eid}")

# =========================================================
# Locate corresponding files in your folder
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"

df_trials_file = [
    f for f in os.listdir(BASE_DIR)
    if f.startswith("df_trials_")
    and subject in f
    and date in f
    and region in f
    and eid in f
]

df_nph_file = [
    f for f in os.listdir(BASE_DIR)
    if f.startswith("df_nph_")
    and subject in f
    and date in f
    and region in f
    and eid in f
]

if not df_trials_file or not df_nph_file:
    print("⚠️ Missing one or both files in folder!")
else:
    df_trials_path = os.path.join(BASE_DIR, df_trials_file[0])
    df_nph_path = os.path.join(BASE_DIR, df_nph_file[0])

    print(f"✅ Found df_trials -> {df_trials_path}")
    print(f"✅ Found df_nph -> {df_nph_path}")

    # Load them
    df_trials = pd.read_csv(df_trials_path)
    df_nph = pd.read_csv(df_nph_path)

    print(f"df_trials shape: {df_trials.shape}")
    print(f"df_nph shape: {df_nph.shape}")

#%% 
""" 
#########################################################################################
# =========================================================
# PLOT CORRECT VS INCORRECT ALIGNED TO EVENT
# change event and peri_event_window
# =========================================================
""" 
from functions import *

time_diffs = df_nph["times"].diff().dropna()
fs = 1/time_diffs.median()
fs

EVENT = "stimOnTrigger_times"
EVENT = "feedback_times"
PERIEVENT_WINDOW = [-1, 2]

photometry_feedback, idx_psth = psth(
    calcium=df_nph.zdFF.values,
    times=df_nph.times.values,
    t_events=df_trials[EVENT].values,
    fs=fs,
    peri_event_window=PERIEVENT_WINDOW
)

n_timepoints = photometry_feedback.shape[0]
time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

plt.figure(figsize=(10, 6))
# Mask for correct vs incorrect/other
mask_correct = df_trials.feedbackType.values == 1
mask_incorrect = ~mask_correct
# Plot trials separately
plt.plot(time_axis, photometry_feedback[:, mask_correct], color='#067bc2', linewidth=0.3, alpha=0.2)
plt.plot(time_axis, photometry_feedback[:, mask_incorrect], color='#ef2b2b', linewidth=0.3, alpha=0.2) 
# --- Mean traces on top ---
mean_correct = np.nanmean(photometry_feedback[:, mask_correct], axis=1)
mean_incorrect = np.nanmean(photometry_feedback[:, mask_incorrect], axis=1)
plt.plot(time_axis, mean_correct, color='#067bc2', linewidth=3, label="Correct (mean)")
plt.plot(time_axis, mean_incorrect, color='#ef2b2b', linewidth=3, label="Incorrect (mean)")
# Event marker
plt.axvline(x=0, color='black', linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("ΔF/F (z-scored)")
plt.title("Peri-feedback PSTH split by feedback type")
plt.ylim(-1.5,3)
plt.show() 

# %%
"""
#########################################################################################
# =========================================================
# PLOT allContrasts ALIGNED TO EVENT
# With SEM shading and option to split by feedbackType
# =========================================================
"""
from functions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIG
# =========================================================
EVENT = "stimOnTrigger_times"
EVENT = "feedback_times"
PERIEVENT_WINDOW = [-1, 2]
split_by_correct = True   # 👈 toggle here (True → split by correct/incorrect; False → average all)
palette = "inferno_r"      # try "viridis", "rocket", "crest", "coolwarm"

# =========================================================
# Compute sampling rate
# =========================================================
time_diffs = df_nph["times"].diff().dropna()
fs = 1 / time_diffs.median()

# =========================================================
# Build PSTH (peri-event calcium)
# =========================================================
photometry_feedback, idx_psth = psth(
    calcium=df_nph.zdFF.values,
    times=df_nph.times.values,
    t_events=df_trials[EVENT].values,
    fs=fs,
    peri_event_window=PERIEVENT_WINDOW
)

n_timepoints = photometry_feedback.shape[0]
time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

# =========================================================
# SPLIT BY CONTRAST LEVELS
# =========================================================
unique_contrasts = np.sort(df_trials["allContrasts"].dropna().unique())
print(f"🎯 Found {len(unique_contrasts)} contrast levels: {unique_contrasts}")

# Define color palette
colors = sns.color_palette(palette, len(unique_contrasts))

plt.figure(figsize=(10, 8))

for i, contrast in enumerate(unique_contrasts):
    mask_contrast = df_trials["allContrasts"] == contrast

    if split_by_correct:
        subsets = {
            "Correct": df_trials["feedbackType"] == 1,
            "Incorrect": df_trials["feedbackType"] != 1
        }
    else:
        subsets = {"All trials": np.ones(len(df_trials), dtype=bool)}

    for label, mask_fb in subsets.items():
        mask = mask_contrast & mask_fb

        if mask.sum() < 3:
            continue  # skip if too few trials

        data = photometry_feedback[:, mask]
        mean_trace = np.nanmean(data, axis=1)
        sem_trace = np.nanstd(data, axis=1) / np.sqrt(np.sum(mask))

        linestyle = '-' if label == "Correct" else '--'
        alpha = 1.0 if label == "Correct" else 0.8

        # --- Plot with SEM shading
        plt.plot(time_axis, mean_trace, color=colors[i],
                 linestyle=linestyle, linewidth=2.5,
                 label=f"{label} — contrast {contrast:.2f}")
        plt.fill_between(time_axis,
                         mean_trace - sem_trace,
                         mean_trace + sem_trace,
                         color=colors[i], alpha=0.25)

# =========================================================
# FINAL FORMATTING
# =========================================================
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("ΔF/F (z-scored)")
title_suffix = "split by correctness" if split_by_correct else "all trials combined"
plt.title(f"Peri-{EVENT} PSTH by contrast ({title_suffix})")
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
# plt.ylim(-1.5, 3)
plt.show()

# %%
# %%
# %%
"""
#########################################################################################
PSTH per trial — aligned at feedback_times = 0
Each line spans the full trial duration relative to feedback
#########################################################################################
"""
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PARAMETERS
# =========================================================
EVENT = "feedback_times"
# EVENT = "feedback_times"

t = df_nph["times"].values
calcium = df_nph["zdFF"].values
feedback_times = df_trials[EVENT].dropna().values
trial_starts = df_trials["intervals_0"].values
trial_ends = df_trials["intervals_1"].values

# Keep only valid feedbacks (not NaN)
valid_mask = ~np.isnan(feedback_times)
feedback_times = feedback_times[valid_mask]
trial_starts = trial_starts[valid_mask]
trial_ends = trial_ends[valid_mask]

# =========================================================
# BUILD TRIAL ALIGNED MATRIX (VARIABLE LENGTH)
# =========================================================
trial_traces = []
time_axes = []

for i, (t0, t_fb, t1) in enumerate(zip(trial_starts, feedback_times, trial_ends)):
    # Extract segment of photometry trace corresponding to this trial
    mask = (t >= t0) & (t <= t1)
    t_segment = t[mask] - t_fb    # align so feedback is 0
    calcium_segment = calcium[mask]
    
    trial_traces.append(calcium_segment)
    time_axes.append(t_segment)

# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(12, 8))

for t_seg, c_seg in zip(time_axes, trial_traces):
    plt.plot(t_seg, c_seg, color='gray', alpha=0.3, linewidth=0.8)

plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel("Time relative to feedback (s)")
plt.ylabel("ΔF/F (z-scored)")
plt.title("ΔF/F trace per trial (aligned at feedback = 0)")
plt.xlim(-2,2.55)

plt.tight_layout()
plt.show()

# %%

# %%
# %%
"""
#########################################################################################
# =========================================================
# HEATMAPS aligned to feedback_times — split by feedbackType
# Trials sorted by index; PSTHs share y-axis
# =========================================================
"""
from functions import *
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
EVENT = "stimOnTrigger_times"
PERIEVENT_WINDOW = [-5, 5]

# =========================================================
# Compute sampling rate
# =========================================================
time_diffs = df_nph["times"].diff().dropna()
fs = 1 / time_diffs.median()

# =========================================================
# Build peri-event ΔF/F matrix
# =========================================================
photometry_feedback, idx_psth = psth(
    calcium=df_nph.zdFF.values,
    times=df_nph.times.values,
    t_events=df_trials[EVENT].dropna().values,
    fs=fs,
    peri_event_window=PERIEVENT_WINDOW
)

n_timepoints, n_trials = photometry_feedback.shape
time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

# =========================================================
# Split trials by correctness
# =========================================================
mask_correct = df_trials["feedbackType"].values == 1
mask_incorrect = ~mask_correct

data_correct = photometry_feedback[:, mask_correct]
data_incorrect = photometry_feedback[:, mask_incorrect]

# =========================================================
# Sort trials by their index (earliest → latest)
# =========================================================
sorted_correct = data_correct[:, np.argsort(np.arange(data_correct.shape[1]))]
sorted_incorrect = data_incorrect[:, np.argsort(np.arange(data_incorrect.shape[1]))]

# =========================================================
# FIGURE
# =========================================================
fig, axes = plt.subplots(
    2, 2,
    figsize=(10, 15),
    gridspec_kw={'height_ratios': [4, 1]},
    sharex=True
)

# Common vmin/vmax for consistent color scale
vmin = np.nanpercentile(photometry_feedback, 5)
vmax = np.nanpercentile(photometry_feedback, 95)

# =========================================================
# HEATMAP — Correct
# =========================================================
axes[0, 0].imshow(
    sorted_correct.T, aspect='auto', origin='lower',
    extent=[time_axis[0], time_axis[-1], 0, sorted_correct.shape[1]],
    cmap='magma', vmin=vmin, vmax=vmax
)
axes[0, 0].axvline(0, color='w', linestyle='--', linewidth=1)
axes[0, 0].set_ylabel("Trials (by index)")
axes[0, 0].set_title("Correct")

# =========================================================
# HEATMAP — Incorrect
# =========================================================
im = axes[0, 1].imshow(
    sorted_incorrect.T, aspect='auto', origin='lower',
    extent=[time_axis[0], time_axis[-1], 0, sorted_incorrect.shape[1]],
    cmap='magma', vmin=vmin, vmax=vmax
)
axes[0, 1].axvline(0, color='w', linestyle='--', linewidth=1)
axes[0, 1].set_title("Incorrect")
axes[0, 1].set_ylabel("")
fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

# =========================================================
# PSTH (mean ± SEM)
# =========================================================
mean_c = np.nanmean(sorted_correct, axis=1)
sem_c = np.nanstd(sorted_correct, axis=1) / np.sqrt(sorted_correct.shape[1])
mean_i = np.nanmean(sorted_incorrect, axis=1)
sem_i = np.nanstd(sorted_incorrect, axis=1) / np.sqrt(sorted_incorrect.shape[1])

# --- Shared Y limits across PSTHs
ymin = min(np.nanmin(mean_c - sem_c), np.nanmin(mean_i - sem_i))
ymax = max(np.nanmax(mean_c + sem_c), np.nanmax(mean_i + sem_i))

# --- Correct
axes[1, 0].plot(time_axis, mean_c, color='#06b5c2', linewidth=2.5)
axes[1, 0].fill_between(time_axis, mean_c - sem_c, mean_c + sem_c,
                        color='#06b5c2', alpha=0.3)
axes[1, 0].axvline(0, color='k', linestyle='--')
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("ΔF/F (z)")
axes[1, 0].set_ylim(ymin, ymax)

# --- Incorrect
axes[1, 1].plot(time_axis, mean_i, color='#ef2b2b', linewidth=2.5)
axes[1, 1].fill_between(time_axis, mean_i - sem_i, mean_i + sem_i,
                        color='#ef2b2b', alpha=0.3)
axes[1, 1].axvline(0, color='k', linestyle='--')
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("ΔF/F (z)")
axes[1, 1].set_ylim(ymin, ymax)

# =========================================================
# FINALIZE
# =========================================================
plt.tight_layout()
plt.show()






#%%
#%%
#%%
# %%
"""
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
######################################################################################### 
""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from functions import *
import sys
sys.path.insert(0, "/home/kceniabougrova/Documents/GitHub/ibl-photometry/src")


df_good_BCW = pd.read_excel("/home/kceniabougrova/Downloads/good_sessions_outputs/df_good_BCW.xlsx")

i = 5
row = df_good_BCW.iloc[i]

# =========================================================
# Extract metadata
# =========================================================
subject = row['subject']
date = str(row['date'])[:10]
region = row['region']
fiber = row['fiber']
eid = row['eid']

print(f"📦 Session {i} — {subject} | {date} | {region} | {fiber} | {eid}")

# =========================================================
# Locate corresponding files in your folder
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"

df_trials_file = [
    f for f in os.listdir(BASE_DIR)
    if f.startswith("df_trials_")
    and subject in f
    and date in f
    and region in f
    and eid in f
]

df_nph_file = [
    f for f in os.listdir(BASE_DIR)
    if f.startswith("df_nph_")
    and subject in f
    and date in f
    and region in f
    and eid in f
]

if not df_trials_file or not df_nph_file:
    print("⚠️ Missing one or both files in folder!")
else:
    df_trials_path = os.path.join(BASE_DIR, df_trials_file[0])
    df_nph_path = os.path.join(BASE_DIR, df_nph_file[0])

    print(f"✅ Found df_trials -> {df_trials_path}")
    print(f"✅ Found df_nph -> {df_nph_path}")

    # Load them
    df_trials = pd.read_csv(df_trials_path)
    df_nph = pd.read_csv(df_nph_path)

    print(f"df_trials shape: {df_trials.shape}")
    print(f"df_nph shape: {df_nph.shape}")

#%% 
""" 
#########################################################################################
# =========================================================
# PLOT CORRECT VS INCORRECT ALIGNED TO EVENT
# change event and peri_event_window
# =========================================================
""" 
from functions import *

time_diffs = df_nph["times"].diff().dropna()
fs = 1/time_diffs.median()
fs

EVENT = "stimOnTrigger_times"
EVENTS = ["stimOnTrigger_times", "feedback_times"]

PERIEVENT_WINDOW = [-1, 2]

for event in EVENTS: 
    photometry_feedback, idx_psth = psth(
        calcium=df_nph.zdFF.values,
        times=df_nph.times.values,
        t_events=df_trials[event].values,
        fs=fs,
        peri_event_window=PERIEVENT_WINDOW
    )

    n_timepoints = photometry_feedback.shape[0]
    time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

    plt.figure(figsize=(8, 6))
    # Mask for correct vs incorrect/other
    mask_correct = df_trials.feedbackType.values == 1
    mask_incorrect = ~mask_correct
    # Plot trials separately
    plt.plot(time_axis, photometry_feedback[:, mask_correct], color='#067bc2', linewidth=0.3, alpha=0.2)
    plt.plot(time_axis, photometry_feedback[:, mask_incorrect], color='#ef2b2b', linewidth=0.3, alpha=0.2) 
    # --- Mean traces on top ---
    mean_correct = np.nanmean(photometry_feedback[:, mask_correct], axis=1)
    mean_incorrect = np.nanmean(photometry_feedback[:, mask_incorrect], axis=1)
    plt.plot(time_axis, mean_correct, color='#067bc2', linewidth=3, label="Correct (mean)")
    plt.plot(time_axis, mean_incorrect, color='#ef2b2b', linewidth=3, label="Incorrect (mean)")
    # Event marker
    plt.axvline(x=0, color='black', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("ΔF/F (z-scored)")
    plt.title("Peri-feedback PSTH split by feedback type")
    plt.ylim(-1,2)
    plt.show()

# %%
"""
#########################################################################################
# =========================================================
# PLOT allContrasts ALIGNED TO EVENT
# With SEM shading and option to split by feedbackType
# =========================================================
"""
from functions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIG
# =========================================================
EVENT = "feedback_times"
PERIEVENT_WINDOW = [-1, 2]
split_by_correct = True   # 👈 toggle here (True → split by correct/incorrect; False → average all)
palette = "inferno_r"      # try "viridis", "rocket", "crest", "coolwarm"

# =========================================================
# Compute sampling rate
# =========================================================
time_diffs = df_nph["times"].diff().dropna()
fs = 1 / time_diffs.median()

# =========================================================
# Build PSTH (peri-event calcium)
# =========================================================
photometry_feedback, idx_psth = psth(
    calcium=df_nph.zdFF.values,
    times=df_nph.times.values,
    t_events=df_trials[EVENT].values,
    fs=fs,
    peri_event_window=PERIEVENT_WINDOW
)

n_timepoints = photometry_feedback.shape[0]
time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

# =========================================================
# SPLIT BY CONTRAST LEVELS
# =========================================================
unique_contrasts = np.sort(df_trials["allContrasts"].dropna().unique())
print(f"🎯 Found {len(unique_contrasts)} contrast levels: {unique_contrasts}")

# Define color palette
colors = sns.color_palette(palette, len(unique_contrasts))

plt.figure(figsize=(8, 6))

for i, contrast in enumerate(unique_contrasts):
    mask_contrast = df_trials["allContrasts"] == contrast

    if split_by_correct:
        subsets = {
            "Correct": df_trials["feedbackType"] == 1,
            "Incorrect": df_trials["feedbackType"] != 1
        }
    else:
        subsets = {"All trials": np.ones(len(df_trials), dtype=bool)}

    for label, mask_fb in subsets.items():
        mask = mask_contrast & mask_fb

        if mask.sum() < 3:
            continue  # skip if too few trials

        data = photometry_feedback[:, mask]
        mean_trace = np.nanmean(data, axis=1)
        sem_trace = np.nanstd(data, axis=1) / np.sqrt(np.sum(mask))

        linestyle = '-' if label == "Correct" else '--'
        alpha = 1.0 if label == "Correct" else 0.8

        # --- Plot with SEM shading
        plt.plot(time_axis, mean_trace, color=colors[i],
                 linestyle=linestyle, linewidth=2.5,
                 label=f"{label} — contrast {contrast:.2f}")
        plt.fill_between(time_axis,
                         mean_trace - sem_trace,
                         mean_trace + sem_trace,
                         color=colors[i], alpha=0.25)

# =========================================================
# FINAL FORMATTING
# =========================================================
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("ΔF/F (z-scored)")
title_suffix = "split by correctness" if split_by_correct else "all trials combined"
plt.title(f"Peri-{EVENT} PSTH by contrast ({title_suffix})")
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
# plt.ylim(-1.5, 3)
plt.show()


# %%
"""
#########################################################################################
PSTH per trial — aligned at feedback_times = 0
Each line spans the full trial duration relative to feedback
"""
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PARAMETERS
# =========================================================
EVENT = "stimOnTrigger_times"
# EVENT = "feedback_times"

t = df_nph["times"].values
calcium = df_nph["zdFF"].values
feedback_times = df_trials[EVENT].dropna().values
trial_starts = df_trials["intervals_0"].values
trial_ends = df_trials["intervals_1"].values

# Keep only valid feedbacks (not NaN)
valid_mask = ~np.isnan(feedback_times)
feedback_times = feedback_times[valid_mask]
trial_starts = trial_starts[valid_mask]
trial_ends = trial_ends[valid_mask]

# =========================================================
# BUILD TRIAL ALIGNED MATRIX (VARIABLE LENGTH)
# =========================================================
trial_traces = []
time_axes = []

for i, (t0, t_fb, t1) in enumerate(zip(trial_starts, feedback_times, trial_ends)):
    # Extract segment of photometry trace corresponding to this trial
    mask = (t >= t0) & (t <= t1)
    t_segment = t[mask] - t_fb    # align so feedback is 0
    calcium_segment = calcium[mask]
    
    trial_traces.append(calcium_segment)
    time_axes.append(t_segment)

# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(12, 8))

for t_seg, c_seg in zip(time_axes, trial_traces):
    plt.plot(t_seg, c_seg, color='gray', alpha=0.3, linewidth=0.8)

plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel("Time relative to feedback (s)")
plt.ylabel("ΔF/F (z-scored)")
plt.title("ΔF/F trace per trial (aligned at feedback = 0)")
plt.xlim(-2,2)

plt.tight_layout()
plt.show()

# %%
# %%
###################################################################################################
# ================================================
# 0) CONFIG
# ================================================
EVENT = "stimOnTrigger_times"
PERIEVENT_WINDOW = [-1, 2]       # seconds around event; change if you want
SMOOTH_WIN = None                # e.g., 5 for small moving average; or None
Z_PER_TRIAL = True               # z-score each trial shape before PCA/cluster
K_RANGE = range(2, 9)            # candidate #clusters for GMM (BIC will choose)
RANDOM_STATE = 7

from functions import psth   # you already have this
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# ================================================
# 1) Build fixed-length trial matrix around feedback
#    -> rows=trials, cols=timepoints
# ================================================
# sampling rate
time_diffs = df_nph["times"].diff().dropna()
fs = 1 / time_diffs.median()

Y, idx_psth = psth(
    calcium=df_nph.zdFF.values,
    times=df_nph.times.values,
    t_events=df_trials[EVENT].values,
    fs=fs,
    peri_event_window=PERIEVENT_WINDOW
)  # Y shape: (T, Ntrials), T=timepoints

time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], Y.shape[0])

# transpose to (n_trials, n_timepoints)
X_ts = Y.T.copy()

# optional smoothing
if SMOOTH_WIN is not None and SMOOTH_WIN > 1:
    import numpy as np
    kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    X_ts = np.array([np.convolve(row, kernel, mode="same") for row in X_ts])

# per-trial z-score (shape only)
if Z_PER_TRIAL:
    X_ts = (X_ts - np.nanmean(X_ts, axis=1, keepdims=True)) / (np.nanstd(X_ts, axis=1, keepdims=True) + 1e-9)

# drop trials with any NaNs (rare)
valid_rows = ~np.isnan(X_ts).any(axis=1)
X_ts = X_ts[valid_rows]
df_trials_valid = df_trials.loc[valid_rows].reset_index(drop=True)

print(f"Matrix for clustering: {X_ts.shape} (trials x timepoints)")

# ================================================
# 2) PCA: keep 95% variance (or min components)
# ================================================
pca = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE)
Z = pca.fit_transform(X_ts)  # (n_trials, n_components)
print(f"PCA -> {Z.shape[1]} comps, explained={pca.explained_variance_ratio_.sum():.3f}")

# ================================================
# 3) GMM: pick #clusters by BIC, soft assignments
# ================================================
bics, gmms = [], []
for k in K_RANGE:
    g = GaussianMixture(
        n_components=k, covariance_type="full",
        random_state=RANDOM_STATE, n_init=5
    ).fit(Z)
    bics.append(g.bic(Z))
    gmms.append(g)

best_idx = int(np.argmin(bics))
best_gmm = gmms[best_idx]
K_best = K_RANGE[best_idx]
print(f"Chosen clusters (BIC): K={K_best}")

labels = best_gmm.predict(Z)                         # (n_trials,)
proba  = best_gmm.predict_proba(Z)                   # (n_trials, K_best)

# build results df
proba_cols = [f"p_cluster_{k}" for k in range(K_best)]
df_clusters = pd.DataFrame({
    "trial_index": np.arange(len(labels)),
    "cluster": labels,
})
df_clusters[proba_cols] = proba
df_clusters = pd.concat([df_trials_valid.reset_index(drop=True), df_clusters], axis=1)

# ================================================
# 4) Visualizations
# ================================================
# 4a) PC scatter
plt.figure(figsize=(7,5))
sc = plt.scatter(Z[:,0], Z[:,1], c=labels, cmap="tab10", s=12, alpha=0.9)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"PCA of trial shapes (K={K_best})")
plt.colorbar(sc, label="Cluster")
plt.tight_layout()
plt.show()

# 4b) Mean ± SEM trace per cluster
plt.figure(figsize=(10,6))
for k in range(K_best):
    mask = labels == k
    if mask.sum() == 0: 
        continue
    mean_k = X_ts[mask].mean(axis=0)
    sem_k  = X_ts[mask].std(axis=0) / np.sqrt(mask.sum())
    plt.plot(time_axis, mean_k, linewidth=2.5, label=f"Cluster {k} (n={mask.sum()})")
    plt.fill_between(time_axis, mean_k - sem_k, mean_k + sem_k, alpha=0.25)
plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel("Time from feedback (s)"); plt.ylabel("ΔF/F (z, per-trial)")
plt.title("Mean ± SEM per cluster")
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.show()

# 4c) Heatmap sorted by cluster then by peak
order = np.lexsort((
    -X_ts.max(axis=1),   # secondary: within cluster, strong first
    labels               # primary: cluster
))
plt.figure(figsize=(7,6))
plt.imshow(X_ts[order,:], aspect='auto', origin='lower',
           extent=[time_axis[0], time_axis[-1], 0, X_ts.shape[0]],
           cmap='magma', vmin=np.percentile(X_ts,5), vmax=np.percentile(X_ts,95))
plt.axvline(0, color='w', linestyle='--', linewidth=1)
plt.xlabel("Time (s)"); plt.ylabel("Trials (cluster-sorted)")
plt.title("Trial heatmap sorted by cluster")
plt.tight_layout()
plt.show()

# ================================================
# 5) Which behavioral variables relate to clusters?
#    Multinomial Logistic Regression + permutation importance
# ================================================
# ---- choose predictors from df_trials_valid (current + previous trial)
X_behav = pd.DataFrame({
    "probLeft":      df_trials_valid["probabilityLeft"].astype(float),
    "contrast":      df_trials_valid["allContrasts"].astype(float),
    "feedback":      df_trials_valid["feedbackType"].astype(float),
    "choice":        df_trials_valid.get("choice", pd.Series(np.nan, index=df_trials_valid.index)).astype(float),
    "rt":            df_trials_valid.get("response_times", pd.Series(np.nan, index=df_trials_valid.index)).astype(float),
})
# add lag-1 versions (previous trial)
for col in ["probLeft", "contrast", "feedback", "choice", "rt"]:
    X_behav[f"{col}_prev1"] = X_behav[col].shift(1)

# align shapes (drop first row for lag)
valid_idx = ~X_behav.isna().any(axis=1)
X_behav = X_behav.loc[valid_idx].reset_index(drop=True)
y = labels[valid_idx.values]  # clusters (int) for the same rows

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_behav.values)

# multinomial logistic regression
clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=2000,
    random_state=RANDOM_STATE
).fit(X_scaled, y)

print(f"Classification accuracy (in-sample): {clf.score(X_scaled, y):.3f}")

# permutation importance to gauge influence
perm = permutation_importance(clf, X_scaled, y, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.DataFrame({
    "feature": X_behav.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)

print("\nTop predictors of cluster membership:")
print(imp.head(12))

# tidy output: add cluster probabilities & IDs back to the trials dataframe
df_clustered = df_trials_valid.copy()
df_clustered["cluster"] = labels
for k, col in enumerate(proba_cols):
    df_clustered[col] = proba[:, k]




# from umap import UMAP
Z_umap = UMAP(n_neighbors=20, min_dist=0.2, random_state=7).fit_transform(Z)
plt.scatter(Z_umap[:,0], Z_umap[:,1], c=labels, cmap='tab10', s=10)
plt.title("UMAP of trial photometry shapes")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ================================================
# Transition matrix between consecutive trials
# ================================================
K = K_best  # number of clusters from GMM
transitions = Counter(zip(labels[:-1], labels[1:]))
P = np.zeros((K, K))
for (a, b), count in transitions.items():
    P[a, b] = count
P = P / P.sum(axis=1, keepdims=True)  # normalize rows to sum=1

plt.figure(figsize=(5, 4))
sns.heatmap(P, annot=True, fmt=".2f", cmap="crest", cbar_kws={"label": "Transition probability"})
plt.xlabel("Next trial cluster")
plt.ylabel("Previous trial cluster")
plt.title(f"Cluster transition matrix (K={K})")
plt.tight_layout()
plt.show()



# %%
import matplotlib.gridspec as gridspec

# --- feature importance dataframe already exists as `imp`

# Create Figure
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[1.1, 1], width_ratios=[1, 1, 0.7])
plt.subplots_adjust(wspace=0.4, hspace=0.3)

# ------------------------------------
# (A) PCA SCATTER
# ------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
sc = ax1.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=10, alpha=0.9)
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
ax1.set_title(f"PCA of trial shapes (K={K_best})")
cbar = plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Cluster ID")

# ------------------------------------
# (B) MEAN ± SEM PSTH PER CLUSTER
# ------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
for k in range(K_best):
    mask = labels == k
    if mask.sum() == 0:
        continue
    mean_k = X_ts[mask].mean(axis=0)
    sem_k = X_ts[mask].std(axis=0) / np.sqrt(mask.sum())
    ax2.plot(time_axis, mean_k, linewidth=2.2, label=f"Cluster {k} (n={mask.sum()})")
    ax2.fill_between(time_axis, mean_k - sem_k, mean_k + sem_k, alpha=0.25)
ax2.axvline(0, color='k', linestyle='--', linewidth=1)
ax2.set_xlabel("Time from feedback (s)")
ax2.set_ylabel("ΔF/F (z, per-trial)")
ax2.set_title("Mean ± SEM per cluster")
ax2.legend(frameon=False, ncol=2, fontsize=8)

# ------------------------------------
# (C) TRIAL HEATMAP (sorted by cluster)
# ------------------------------------
ax3 = fig.add_subplot(gs[1, 0])
order = np.lexsort((-X_ts.max(axis=1), labels))
im = ax3.imshow(X_ts[order, :], aspect='auto', origin='lower',
                extent=[time_axis[0], time_axis[-1], 0, X_ts.shape[0]],
                cmap='magma',
                vmin=np.percentile(X_ts, 5), vmax=np.percentile(X_ts, 95))
ax3.axvline(0, color='w', linestyle='--', linewidth=1)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Trials (sorted)")
ax3.set_title("Trial heatmap by cluster")
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

# ------------------------------------
# (D) CLUSTER TRANSITION MATRIX
# ------------------------------------
ax4 = fig.add_subplot(gs[1, 1])
sns.heatmap(P, ax=ax4, annot=True, fmt=".2f", cmap="crest",
            cbar_kws={"label": "Transition probability"})
ax4.set_xlabel("Next trial cluster")
ax4.set_ylabel("Previous trial cluster")
ax4.set_title("Cluster transition matrix")

# ------------------------------------
# (E) FEATURE IMPORTANCES
# ------------------------------------
ax5 = fig.add_subplot(gs[:, 2])
top_imp = imp.head(10)[::-1]
ax5.barh(top_imp["feature"], top_imp["importance_mean"], xerr=top_imp["importance_std"], color='gray')
ax5.set_xlabel("Permutation importance")
ax5.set_title("Top behavioral predictors")

plt.tight_layout()
plt.show()



# %%
"""
#########################################################################################
#########################################################################################
#########################################################################################
================================================================================
Aggregate ΔF/F responses across all sessions for one subject
================================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions import psth

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
SUBJECT = "ZFM-04026"     # 👈 choose subject
window_baseline = [-0.1, 0]
window_response = [0.01, 0.5]

# =========================================================
# Load metadata table
# =========================================================
df_good_BCW = pd.read_excel(os.path.join(BASE_DIR, "df_good_BCW.xlsx"))
sessions_subj = df_good_BCW[df_good_BCW["subject"] == SUBJECT]
print(f"📦 Found {len(sessions_subj)} sessions for {SUBJECT}")

# =========================================================
# Helper functions
# =========================================================
def mean_sem(y):
    return np.nanmean(y), np.nanstd(y) / np.sqrt(np.sum(~np.isnan(y)))

def compute_response(df_trials, df_nph, event_col, window_baseline, window_response, fs):
    """Baseline-corrected mean ΔF/F during response window"""
    Y, idx_psth = psth(
        calcium=df_nph.zdFF.values,
        times=df_nph.times.values,
        t_events=df_trials[event_col].values,
        fs=fs,
        peri_event_window=[window_baseline[0], window_response[1]]
    )
    time_axis = np.linspace(window_baseline[0], window_response[1], Y.shape[0])
    baseline_mask = (time_axis >= window_baseline[0]) & (time_axis < window_baseline[1])
    response_mask = (time_axis >= window_response[0]) & (time_axis < window_response[1])
    baseline = np.nanmean(Y[baseline_mask, :], axis=0)
    Y_norm = Y - baseline
    response = np.nanmean(Y_norm[response_mask, :], axis=0)
    return response

# =========================================================
# Loop over all sessions for this mouse
# =========================================================
all_trials = []  # to concatenate trials from all sessions

for _, row in sessions_subj.iterrows():
    date = str(row["date"])[:10]
    region = str(row["region"])
    eid = str(row["eid"])

    df_trials_file = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith("df_trials_")
        and SUBJECT in f and date in f and region in f and eid in f
    ]
    df_nph_file = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith("df_nph_")
        and SUBJECT in f and date in f and region in f and eid in f
    ]

    if not df_trials_file or not df_nph_file:
        print(f"⚠️ Skipping {date} — files missing.")
        continue

    df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
    df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

    # Sampling rate
    fs = 1 / df_nph["times"].diff().dropna().median()

    # Compute responses for each event
    stim_response = compute_response(
        df_trials, df_nph,
        event_col="stimOnTrigger_times",
        window_baseline=window_baseline,
        window_response=window_response,
        fs=fs
    )
    fb_response = compute_response(
        df_trials, df_nph,
        event_col="feedback_times",
        window_baseline=window_baseline,
        window_response=window_response,
        fs=fs
    )

    df_trials["stim_resp"] = stim_response
    df_trials["fb_resp"] = fb_response
    df_trials["session_date"] = date
    all_trials.append(df_trials)

# Merge all trials
df_all = pd.concat(all_trials, ignore_index=True)
print(f"✅ Combined {len(df_all)} trials from {len(all_trials)} sessions.")

# =========================================================
# Aggregate per contrast and probLeft
# =========================================================
unique_contrasts = np.sort(df_all["allSContrasts"].dropna().unique())
prob_levels = sorted(df_all["probabilityLeft"].dropna().unique())

# Colors for probabilityLeft
prob_palette = sns.color_palette("tab10", len(prob_levels))
prob_color_map = dict(zip(prob_levels, prob_palette))

# =========================================================
# Function to compute mean/SEM matrix per condition
# =========================================================
def aggregate_by(df, resp_col, feedback_filter=None):
    means, sems = {}, {}
    for p in prob_levels:
        means[p], sems[p] = [], []
        for c in unique_contrasts:
            mask = (df["allSContrasts"] == c) & (df["probabilityLeft"] == p)
            if feedback_filter is not None:
                mask &= feedback_filter
            if mask.sum() < 3:
                means[p].append(np.nan)
                sems[p].append(np.nan)
                continue
            m, s = mean_sem(df.loc[mask, resp_col])
            means[p].append(m); sems[p].append(s)
    return means, sems

stim_means, stim_sems = aggregate_by(df_all, "stim_resp")
fb_means_corr, fb_sems_corr = aggregate_by(df_all, "fb_resp", df_all["feedbackType"] == 1)
fb_means_inc, fb_sems_inc = aggregate_by(df_all, "fb_resp", df_all["feedbackType"] != 1)

# =========================================================
# Helper for compact x-axis with "..." gap
# =========================================================
def spaced_contrasts(contrasts):
    x = contrasts.copy().astype(float)
    mid_mask = (x > -0.75) & (x < 0.75)
    mid = x[mid_mask]
    extremes = x[~mid_mask]
    x_positions = np.linspace(-0.6, 0.6, len(mid))
    new_x = []
    for val in x:
        if val in extremes:
            new_x.append(np.sign(val) * 0.8)
        else:
            new_x.append(x_positions[np.argwhere(mid == val)[0, 0]])
    return np.array(new_x)

x_positions = spaced_contrasts(unique_contrasts)

# =========================================================
# PLOTS
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(8, 5), sharey=True)
titles = ["After stimulus", "Feedback (correct)", "Feedback (incorrect)"]
datasets = [(stim_means, stim_sems), (fb_means_corr, fb_sems_corr), (fb_means_inc, fb_sems_inc)]

for ax, (title, (means, sems)) in zip(axes, zip(titles, datasets)):
    for p in prob_levels:
        ax.errorbar(
            x_positions, means[p], yerr=sems[p], fmt="-o",
            color=prob_color_map[p], label=f"pLeft={p:.1f}", alpha=0.9
        )
    ax.set_title(title)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x_positions)
    xtick_labels = [f"{c:.2f}" if abs(c) < 1 else f"{int(c)}" for c in unique_contrasts]
    # Insert ellipsis if needed
    if np.any(np.abs(unique_contrasts) == 1):
        idx = np.where(np.abs(unique_contrasts) == np.min(np.abs(unique_contrasts[np.abs(unique_contrasts) > 0.25])))[0][-1]
        xtick_labels.insert(idx + 1, "...")
        ax.set_xticks(np.insert(x_positions, idx + 1, 0.7))
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Signed contrast")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title == "After stimulus":
        ax.set_ylabel("ΔF/F (baseline-subtracted)")

axes[0].legend(frameon=False, title="p(Left)", fontsize=8)
plt.tight_layout()
plt.show()


























# %%
"""
#########################################################################################
#########################################################################################
#########################################################################################
10102025
this code plots all sessions from 1 subject - TO CHECK!!!!! 
"""

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# =========================================================
# SETTINGS
# =========================================================
EVENTS = ["stimOnTrigger_times", "feedback_times"]
PERIEVENT_WINDOW = [-1, 2]
cmap = cm.get_cmap("inferno_r")



subject = "ZFM-04019"   
df_mouse = df_good_BCW[df_good_BCW["subject"] == subject].reset_index(drop=True)
print(f"🐭 Found {len(df_mouse)} sessions for {subject}")

# =========================================================
# Load and concatenate all sessions
# =========================================================
all_trials = []
all_nph = []

time_offset = 0  # initialize outside the loop

for idx, row in df_mouse.iterrows():
    date = str(row["date"])[:10]
    region = row["region"]
    eid = row["eid"]

    # Locate corresponding files
    df_trials_file = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith("df_trials_")
        and subject in f
        and date in f
        and region in f
        and eid in f
    ]
    df_nph_file = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith("df_nph_")
        and subject in f
        and date in f
        and region in f
        and eid in f
    ]

    if not df_trials_file or not df_nph_file:
        print(f"⚠️ Missing files for {subject} | {date} | {region} | {eid}")
        continue

    # Load
    df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
    df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

    print(f"✅ Loaded {date} | {region} | {eid}")
    print(f"   df_trials: {df_trials.shape}, df_nph: {df_nph.shape}")
    
    # Apply per-session offset so times are unique
    df_nph["times"] += time_offset
    for ev in ["stimOnTrigger_times", "feedback_times"]:
        if ev in df_trials.columns:
            df_trials[ev] += time_offset

    # Update cumulative offset (add small gap between sessions)
    time_offset += df_nph["times"].iloc[-1] + 5


    # Store with session identifier
    df_trials["session_id"] = f"{date}_{region}_{eid}"
    df_nph["session_id"] = f"{date}_{region}_{eid}"

    all_trials.append(df_trials)
    all_nph.append(df_nph)

# Concatenate all sessions
if not all_trials or not all_nph:
    raise ValueError("No valid sessions found for this subject.")

df_trials_all = pd.concat(all_trials, ignore_index=True)
df_nph_all = pd.concat(all_nph, ignore_index=True)

print(f"📊 Combined df_trials_all shape: {df_trials_all.shape}")
print(f"📈 Combined df_nph_all shape: {df_nph_all.shape}")

# =========================================================
# Compute sampling frequency (fs)
# =========================================================
time_diffs = df_nph_all["times"].diff().dropna()
fs = 1 / time_diffs.median()
print(f"⏱ Sampling rate: {fs:.2f} Hz")





# =========================================================
# LOOP THROUGH EVENTS
# =========================================================
for event in EVENTS:
    print(f"\n🎬 Processing {event}...")

    # Get unique contrast levels (sorted)
    contrasts = np.sort(df_trials_all["allContrasts"].dropna().unique())
    colors = cmap(np.linspace(0, 1, len(contrasts)))

    plt.figure(figsize=(8, 6))

    for i, contrast in enumerate(contrasts):
        mask = df_trials_all["allContrasts"] == contrast

        # Compute PSTH only for these trials
        photometry_event, idx_psth = psth(
            calcium=df_nph_all["zdFF"].values,
            times=df_nph_all["times"].values,
            t_events=df_trials_all.loc[mask, event].values,
            fs=fs,
            peri_event_window=PERIEVENT_WINDOW
        )


        # Mean trace and error
        mean_trace = np.nanmean(photometry_event, axis=1)
        sem_trace = np.nanstd(photometry_event, axis=1) / np.sqrt(photometry_event.shape[1])

        n_timepoints = photometry_event.shape[0]
        time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

        # Plot mean + SEM shadow
        plt.plot(time_axis, mean_trace, color=colors[i], lw=2.5, label=f"{contrast:.2f}")
        plt.fill_between(time_axis,
                         mean_trace - sem_trace,
                         mean_trace + sem_trace,
                         color=colors[i],
                         alpha=0.25)

    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("ΔF/F (z-scored)")
    plt.title(f"{subject} — Combined sessions — {event}")
    plt.legend(title="allContrasts", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

# %%
""" same - to check!!!!! """
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# =========================================================
# SETTINGS
# =========================================================
EVENTS = ["stimOnTrigger_times", "feedback_times"]
PERIEVENT_WINDOW = [-1, 2]
cmap = cm.get_cmap("inferno_r")

subject = "ZFM-04022"
df_mouse = df_good_BCW[df_good_BCW["subject"] == subject].reset_index(drop=True)
print(f"🐭 Found {len(df_mouse)} sessions for {subject}")

# =========================================================
# Load and concatenate all sessions
# =========================================================
all_trials = []
all_nph = []
time_offset = 0  # initialize outside the loop

for idx, row in df_mouse.iterrows():
    date = str(row["date"])[:10]
    region = row["region"]
    eid = row["eid"]

    # Locate corresponding files
    df_trials_file = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith("df_trials_")
        and subject in f
        and date in f
        and region in f
        and eid in f
    ]
    df_nph_file = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith("df_nph_")
        and subject in f
        and date in f
        and region in f
        and eid in f
    ]

    if not df_trials_file or not df_nph_file:
        print(f"⚠️ Missing files for {subject} | {date} | {region} | {eid}")
        continue

    # Load
    df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
    df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

    print(f"✅ Loaded {date} | {region} | {eid}")
    print(f"   df_trials: {df_trials.shape}, df_nph: {df_nph.shape}")

    # Apply per-session offset so times are unique
    df_nph["times"] += time_offset
    for ev in ["stimOnTrigger_times", "feedback_times"]:
        if ev in df_trials.columns:
            df_trials[ev] += time_offset

    # Update cumulative offset (add small gap between sessions)
    time_offset += df_nph["times"].iloc[-1] + 5

    # Store with session identifier
    df_trials["session_id"] = f"{date}_{region}_{eid}"
    df_nph["session_id"] = f"{date}_{region}_{eid}"

    all_trials.append(df_trials)
    all_nph.append(df_nph)

# Concatenate all sessions
if not all_trials or not all_nph:
    raise ValueError("No valid sessions found for this subject.")

df_trials_all = pd.concat(all_trials, ignore_index=True)
df_nph_all = pd.concat(all_nph, ignore_index=True)

print(f"📊 Combined df_trials_all shape: {df_trials_all.shape}")
print(f"📈 Combined df_nph_all shape: {df_nph_all.shape}")

# =========================================================
# Compute sampling frequency (fs)
# =========================================================
time_diffs = df_nph_all["times"].diff().dropna()
fs = 1 / time_diffs.median()
print(f"⏱ Sampling rate: {fs:.2f} Hz")

# =========================================================
# PLOT: 3 SUBPLOTS (stimOn, feedback correct, feedback incorrect)
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
titles = ["Stimulus Onset", "Feedback (Correct)", "Feedback (Incorrect)"]
conditions = [
    {"event": "stimOnTrigger_times", "mask": np.ones(len(df_trials_all), dtype=bool)},
    {"event": "feedback_times", "mask": df_trials_all["feedbackType"] == 1},
    {"event": "feedback_times", "mask": df_trials_all["feedbackType"] != 1},
]

for ax, title, cond in zip(axes, titles, conditions):
    event = cond["event"]
    mask_trials = cond["mask"]

    # Get unique contrast levels
    contrasts = np.sort(df_trials_all["allContrasts"].dropna().unique())
    colors = cmap(np.linspace(0, 1, len(contrasts)))

    for i, contrast in enumerate(contrasts):
        mask_contrast = mask_trials & (df_trials_all["allContrasts"] == contrast)

        # Compute PSTH only for these trials
        photometry_event, idx_psth = psth(
            calcium=df_nph_all["zdFF"].values,
            times=df_nph_all["times"].values,
            t_events=df_trials_all.loc[mask_contrast, event].values,
            fs=fs,
            peri_event_window=PERIEVENT_WINDOW
        )

        if photometry_event.size == 0:
            continue

        # Compute mean ± SEM
        mean_trace = np.nanmean(photometry_event, axis=1)
        sem_trace = np.nanstd(photometry_event, axis=1) / np.sqrt(photometry_event.shape[1])

        n_timepoints = photometry_event.shape[0]
        time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

        # Plot mean + shadow
        ax.plot(time_axis, mean_trace, color=colors[i], lw=2.5, label=f"{contrast:.2f}")
        ax.fill_between(time_axis,
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        color=colors[i],
                        alpha=0.25)

    # Common formatting
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)

axes[0].set_ylabel("ΔF/F (z-scored)")
axes[0].legend(title="allContrasts", bbox_to_anchor=(1.05, 1), loc="upper left")

fig.suptitle(f"{subject} — Combined sessions — Mean ± SEM", fontsize=14, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
""" same, but now loops through all the unique mice and fibers """ 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# =========================================================
# SETTINGS
# =========================================================
EVENTS = ["stimOnTrigger_times", "feedback_times"]
PERIEVENT_WINDOW = [-1, 2]
cmap = cm.get_cmap("inferno_r")

# Load metadata
df_good_BCW = pd.read_excel(os.path.join(BASE_DIR, "df_good_BCW.xlsx"))

# # Get all subjects present
# subjects = df_good_BCW["subject"].unique()
# print(f"🐭 Found {len(subjects)} unique subjects: {', '.join(subjects)}")

# # =========================================================
# # LOOP THROUGH SUBJECTS
# # =========================================================
# for subject in subjects:
#     print(f"\n===============================")
#     print(f"🐭 Processing subject: {subject}")
#     print(f"===============================")

#     df_mouse = df_good_BCW[df_good_BCW["subject"] == subject].reset_index(drop=True)
#     print(f"📦 {len(df_mouse)} sessions for {subject}")
""" ORRRRRR """
# =========================================================
# GROUP BY UNIQUE SUBJECT × REGION (fiber)
# =========================================================
grouped = df_good_BCW.groupby(["subject", "fiber"])
print(f"🔎 Found {len(grouped)} unique subject × unique fiber")

for (subject, fiber), df_mouse in grouped:
    print(f"\n===============================")
    print(f"🐭 Processing {subject} — {fiber}")
    print(f"===============================")


    df_mouse = df_mouse.reset_index(drop=True)
    print(f"📦 {len(df_mouse)} sessions for {subject} ({region})")


    # =========================================================
    # Load and concatenate all sessions
    # =========================================================
    all_trials = []
    all_nph = []
    time_offset = 0

    for idx, row in df_mouse.iterrows():
        date = str(row["date"])[:10]
        region = row["region"]
        eid = row["eid"]

        # Locate corresponding files
        df_trials_file = [
            f for f in os.listdir(BASE_DIR)
            if f.startswith("df_trials_")
            and subject in f
            and date in f
            and region in f
            and eid in f
        ]
        df_nph_file = [
            f for f in os.listdir(BASE_DIR)
            if f.startswith("df_nph_")
            and subject in f
            and date in f
            and region in f
            and eid in f
        ]

        if not df_trials_file or not df_nph_file:
            print(f"⚠️ Missing files for {subject} | {date} | {region} | {eid}")
            continue

        # Load
        df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
        df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

        # Apply per-session offset so times are unique
        df_nph["times"] += time_offset
        for ev in ["stimOnTrigger_times", "feedback_times"]:
            if ev in df_trials.columns:
                df_trials[ev] += time_offset

        # Update cumulative offset (+ small gap)
        time_offset += df_nph["times"].iloc[-1] + 5

        # Store with session identifier
        df_trials["session_id"] = f"{date}_{region}_{eid}"
        df_nph["session_id"] = f"{date}_{region}_{eid}"

        all_trials.append(df_trials)
        all_nph.append(df_nph)

    # Skip if subject has no valid sessions
    if not all_trials or not all_nph:
        print(f"⚠️ No valid sessions found for {subject}")
        continue

    # Concatenate
    df_trials_all = pd.concat(all_trials, ignore_index=True)
    df_nph_all = pd.concat(all_nph, ignore_index=True)

    print(f"📊 Combined df_trials_all shape: {df_trials_all.shape}")
    print(f"📈 Combined df_nph_all shape: {df_nph_all.shape}")

    # =========================================================
    # Compute sampling frequency
    # =========================================================
    time_diffs = df_nph_all["times"].diff().dropna()
    fs = 1 / time_diffs.median()
    print(f"⏱ Sampling rate: {fs:.2f} Hz")

    # =========================================================
    # PLOT: 3 SUBPLOTS (stimOn, feedback correct, feedback incorrect)
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    titles = ["Stimulus Onset", "Feedback (Correct)", "Feedback (Incorrect)"]
    conditions = [
        {"event": "stimOnTrigger_times", "mask": np.ones(len(df_trials_all), dtype=bool)},
        {"event": "feedback_times", "mask": df_trials_all["feedbackType"] == 1},
        {"event": "feedback_times", "mask": df_trials_all["feedbackType"] != 1},
    ]

    for ax, title, cond in zip(axes, titles, conditions):
        event = cond["event"]
        mask_trials = cond["mask"]

        contrasts = np.sort(df_trials_all["allContrasts"].dropna().unique())
        colors = cmap(np.linspace(0, 1, len(contrasts)))

        for i, contrast in enumerate(contrasts):
            mask_contrast = mask_trials & (df_trials_all["allContrasts"] == contrast)

            # Compute PSTH
            photometry_event, idx_psth = psth(
                calcium=df_nph_all["zdFF"].values,
                times=df_nph_all["times"].values,
                t_events=df_trials_all.loc[mask_contrast, event].values,
                fs=fs,
                peri_event_window=PERIEVENT_WINDOW
            )

            if photometry_event.size == 0:
                continue

            mean_trace = np.nanmean(photometry_event, axis=1)
            sem_trace = np.nanstd(photometry_event, axis=1) / np.sqrt(photometry_event.shape[1])

            n_timepoints = photometry_event.shape[0]
            time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

            ax.plot(time_axis, mean_trace, color=colors[i], lw=2.5, label=f"{contrast:.2f}")
            ax.fill_between(time_axis,
                            mean_trace - sem_trace,
                            mean_trace + sem_trace,
                            color=colors[i],
                            alpha=0.25)

        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_title(title)

    axes[0].set_ylabel("ΔF/F (z-scored)")
    axes[0].legend(title="allContrasts", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.suptitle(f"{subject} — Combined sessions — Mean ± SEM", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()





#%%
"""
3 subplots - signed contrasts - loops through all the mice 
""" 
# %%
""" same, but now loops through all the unique mice and fibers """ 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# =========================================================
# SETTINGS
# =========================================================
EVENTS = ["stimOnTrigger_times", "feedback_times"]
PERIEVENT_WINDOW = [-1, 2]
cmap = cm.get_cmap("inferno_r")

# Load metadata
df_good_BCW = pd.read_excel(os.path.join(BASE_DIR, "df_good_BCW.xlsx"))

# Get all subjects present
subjects = df_good_BCW["subject"].unique()
print(f"🐭 Found {len(subjects)} unique subjects: {', '.join(subjects)}")

# =========================================================
# LOOP THROUGH SUBJECTS
# =========================================================
for subject in subjects:
    print(f"\n===============================")
    print(f"🐭 Processing subject: {subject}")
    print(f"===============================")

    df_mouse = df_good_BCW[df_good_BCW["subject"] == subject].reset_index(drop=True)
    print(f"📦 {len(df_mouse)} sessions for {subject}")
    """ ORRRRRR """
    # # =========================================================
    # # GROUP BY UNIQUE SUBJECT × REGION (fiber)
    # # =========================================================
    # grouped = df_good_BCW.groupby(["subject", "fiber"])
    # print(f"🔎 Found {len(grouped)} unique subject × unique fiber")

    # for (subject, fiber), df_mouse in grouped:
    #     print(f"\n===============================")
    #     print(f"🐭 Processing {subject} — {fiber}")
    #     print(f"===============================")


    #     df_mouse = df_mouse.reset_index(drop=True)
    #     print(f"📦 {len(df_mouse)} sessions for {subject} ({region})")


    # =========================================================
    # Load and concatenate all sessions
    # =========================================================
    all_trials = []
    all_nph = []
    time_offset = 0

    for idx, row in df_mouse.iterrows():
        date = str(row["date"])[:10]
        region = row["region"]
        eid = row["eid"]

        # Locate corresponding files
        df_trials_file = [
            f for f in os.listdir(BASE_DIR)
            if f.startswith("df_trials_")
            and subject in f
            and date in f
            and region in f
            and eid in f
        ]
        df_nph_file = [
            f for f in os.listdir(BASE_DIR)
            if f.startswith("df_nph_")
            and subject in f
            and date in f
            and region in f
            and eid in f
        ]

        if not df_trials_file or not df_nph_file:
            print(f"⚠️ Missing files for {subject} | {date} | {region} | {eid}")
            continue

        # Load
        df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
        df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

        # Apply per-session offset so times are unique
        df_nph["times"] += time_offset
        for ev in ["stimOnTrigger_times", "feedback_times"]:
            if ev in df_trials.columns:
                df_trials[ev] += time_offset

        # Update cumulative offset (+ small gap)
        time_offset += df_nph["times"].iloc[-1] + 5

        # Store with session identifier
        df_trials["session_id"] = f"{date}_{region}_{eid}"
        df_nph["session_id"] = f"{date}_{region}_{eid}"

        all_trials.append(df_trials)
        all_nph.append(df_nph)

    # Skip if subject has no valid sessions
    if not all_trials or not all_nph:
        print(f"⚠️ No valid sessions found for {subject}")
        continue

    # Concatenate
    df_trials_all = pd.concat(all_trials, ignore_index=True)
    df_nph_all = pd.concat(all_nph, ignore_index=True)

    print(f"📊 Combined df_trials_all shape: {df_trials_all.shape}")
    print(f"📈 Combined df_nph_all shape: {df_nph_all.shape}")

    # =========================================================
    # Compute sampling frequency
    # =========================================================
    time_diffs = df_nph_all["times"].diff().dropna()
    fs = 1 / time_diffs.median()
    print(f"⏱ Sampling rate: {fs:.2f} Hz")

    # =========================================================
    # PLOT: 3 SUBPLOTS (stimOn, feedback correct, feedback incorrect)
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    titles = ["Stimulus Onset", "Feedback (Correct)", "Feedback (Incorrect)"]
    conditions = [
        {"event": "stimOnTrigger_times", "mask": np.ones(len(df_trials_all), dtype=bool)},
        {"event": "feedback_times", "mask": df_trials_all["feedbackType"] == 1},
        {"event": "feedback_times", "mask": df_trials_all["feedbackType"] != 1},
    ]

    for ax, title, cond in zip(axes, titles, conditions):
        event = cond["event"]
        mask_trials = cond["mask"]

        contrasts = np.sort(df_trials_all["allSContrasts"].dropna().unique())

        # Define colors: same hue for positive/negative, darker for stronger contrasts
        max_contrast = np.max(np.abs(contrasts))
        abs_contrasts = np.abs(contrasts)
        # Normalize so 0 → light, 1 → dark
        normed = abs_contrasts / max_contrast

        # Use a perceptually uniform colormap (you can keep inferno_r if you like)
        base_cmap = cm.get_cmap("inferno_r")

        colors = []
        for c in contrasts:
            # Darker for larger |contrast|
            intensity = 0.2 + 0.8 * (abs(c) / max_contrast)  # keeps some range
            # Positive and negative get same hue
            color = base_cmap(intensity)
            colors.append(color)


        for i, contrast in enumerate(contrasts):
            mask_contrast = mask_trials & (df_trials_all["allSContrasts"] == contrast)

            # Compute PSTH
            photometry_event, idx_psth = psth(
                calcium=df_nph_all["zdFF"].values,
                times=df_nph_all["times"].values,
                t_events=df_trials_all.loc[mask_contrast, event].values,
                fs=fs,
                peri_event_window=PERIEVENT_WINDOW
            )

            if photometry_event.size == 0:
                continue

            mean_trace = np.nanmean(photometry_event, axis=1)
            sem_trace = np.nanstd(photometry_event, axis=1) / np.sqrt(photometry_event.shape[1])

            n_timepoints = photometry_event.shape[0]
            time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], n_timepoints)

            ax.plot(time_axis, mean_trace, color=colors[i], lw=2.5, label=f"{contrast:.2f}")
            ax.fill_between(time_axis,
                            mean_trace - sem_trace,
                            mean_trace + sem_trace,
                            color=colors[i],
                            alpha=0.25)

        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_title(title)

    axes[0].set_ylabel("ΔF/F (z-scored)")
    axes[0].legend(title="allContrasts", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.suptitle(f"{subject} — Combined sessions — Mean ± SEM", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

