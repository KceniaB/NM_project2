#%% 
""" 
31 0ctober 2025
"""

# ##################################
# IMPORTS
# ##################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from functions import *
import sys
sys.path.insert(0, "/home/kceniabougrova/Documents/GitHub/ibl-photometry/src")

from one.api import ONE 
one = ONE() 


# ##################################
# FUNCTIONS
# ##################################
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
    eid = sessions_list.loc[row, "eid"]
    subject = sessions_list.loc[row, "subject"]
    date = sessions_list.loc[row, "date"]
    region = sessions_list.loc[row, "region"]
    nph_file_path = sessions_list.loc[row, "photometry_path_a"]
    if pd.isna(nph_file_path) or not nph_file_path:  # if empty or NaN
        nph_file_path = sessions_list.loc[row, "photometry_path_b"]
    nph_bnc_path = sessions_list.loc[row, "digital_inputs_path"]
    eid2 = get_eid(subject, date)
    if eid2 != eid:
        print(f"⚠️ Different eids for {subject} on {date}: {eid} vs {eid2}")
    print(subject, date, eid)
    return eid, subject, date, region, nph_file_path, nph_bnc_path

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

def load_trials_updated_for_sessions_with_task00(eid): 
    # trials = one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf')
    # trials = one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf/task_00')
    trials = one.load_object(eid, 'trials', collection='alf')

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




#%% 
# ##################################
# LOAD DATA FROM TABLE
# have: table path with these columns: 
#   eid, subject, date, region, photometry_path_a, photometry_path_b, digital_inputs_path
# ##################################
table_path = '/home/kceniabougrova/Downloads/new_sessions_Aug2025 - all_together_tab2_Oct2025.csv'
sessions_list = pd.read_csv(table_path)

i = 1000 #select the session here #DA
# i = 1626

# just in case the path was not yet updated (should already be - update from end of October2025)
old_prefix = '/mnt/h0/kb/'
new_prefix = '/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/'

print(f"\n====================")
print(f"Processing session {i}: {sessions_list['subject'][i]} {sessions_list['date'][i]} {sessions_list['region'][i]}")
print("====================")

# =========================================================
# Retrieve all session info 
# =========================================================
eid, subject, date, region, nph_file_path, nph_bnc_path = select_one_session(
    sessions_list, row=i)

# Fix paths
nph_file_path = nph_file_path.replace(old_prefix, new_prefix)
nph_bnc_path = nph_bnc_path.replace(old_prefix, new_prefix)

# =========================================================
# Load data
# =========================================================
df_nph = pd.read_csv(nph_file_path)
df_bnc = pd.read_csv(nph_bnc_path)

# Load behavior
try:
    df_trials, subject, session_date = load_trials_updated(eid)
    if df_trials is None or df_trials.empty:
        raise ValueError("empty df_trials")
except Exception:
    df_trials, subject, session_date = load_trials_updated_for_sessions_with_task00(eid)
# # Save df_trials
# save_trials_path = f"/home/kceniabougrova/Downloads/good_sessions_outputs/df_trials_{idx}_{subject}_{date}_{region}_{eid}.csv"
# df_trials.to_csv(save_trials_path, index=False)
# print(f"💾 Saved df_trials -> {save_trials_path}")

#%% 
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

# add new times
df_nph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_nph["Timestamp"])

# Crop nph to behavior session 
session_start = df_trials.intervals_0.values[0] - 10
session_end = df_trials.intervals_1.values[-1] + 10
df_nph = df_nph[
    (df_nph['bpod_frame_times'] >= session_start) &
    (df_nph['bpod_frame_times'] <= session_end)
].reset_index(drop=True)

print(f"✅ Sync and crop done for {subject} {date}")

# =========================================================
# Photometry signal preprocessing
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
# save_nph_path = f"/home/kceniabougrova/Downloads/good_sessions_outputs/df_nph_{idx}_{subject}_{date}_{region}_{eid}.csv"
# df_nph.to_csv(save_nph_path, index=False)
# print(f"💾 Saved df_nph -> {save_nph_path}")







#%% 
"""
# =========================================================================================
At this moment you should have df_nph and df_trials loaded and preprocessed, ready to plot
Important columns in df_nph: 'times', 'zdFF'
Times are already in the same clock
# =========================================================================================
"""
#%%
# =========================================================
#  PSTH plot (Peri-feedback activity)
# =========================================================
PLOT = True 

if PLOT:
    try:
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
        plt.figure(figsize=(8, 6), dpi=300)
        mask_correct = df_trials.feedbackType.values == 1
        mask_incorrect = ~mask_correct

        # Plot all trials (optional)
        plt.plot(time_axis, photometry_feedback[:, mask_correct], color='#0077b6', alpha=0.1, linewidth=0.5)
        plt.plot(time_axis, photometry_feedback[:, mask_incorrect], color='red', alpha=0.1, linewidth=0.5)

        # Mean ± SEM
        mean_correct = np.nanmean(photometry_feedback[:, mask_correct], axis=1)
        mean_incorrect = np.nanmean(photometry_feedback[:, mask_incorrect], axis=1)
        sem_correct = np.nanstd(photometry_feedback[:, mask_correct], axis=1) / np.sqrt(mask_correct.sum())
        sem_incorrect = np.nanstd(photometry_feedback[:, mask_incorrect], axis=1) / np.sqrt(mask_incorrect.sum())

        plt.plot(time_axis, mean_correct, color='#0077b6', linewidth=2.5, label='Correct')
        plt.fill_between(time_axis, mean_correct - sem_correct, mean_correct + sem_correct,
                        color='#0077b6', alpha=0.3)
        plt.plot(time_axis, mean_incorrect, color='red', linewidth=2.5, label='Incorrect')
        plt.fill_between(time_axis, mean_incorrect - sem_incorrect, mean_incorrect + sem_incorrect,
                        color='red', alpha=0.3)

        plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
        plt.title(f"{i} - PSTH peri-{EVENT} — {subject} {region} {date}")
        plt.xlabel("Time (s)")
        plt.ylabel("ΔF/F (z-scored)")
        plt.legend(frameon=False)
        plt.ylim(-2,3)
        plt.tight_layout() 
        # --- Save plot
        # plot_path = f"/home/kceniabougrova/Downloads/good_sessions_outputs/plot_{idx}_{subject}_{date}_{region}_{eid}.png"
        # plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        # print(f"🖼️  Saved plot -> {plot_path}")

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show(block=False)
        plt.pause(0.1)  # ensures plot shows before next loop iteration

    except Exception as e:
        print(f"⚠️ Could not plot PSTH for {subject} {date} {region}: {e}")



# %%
