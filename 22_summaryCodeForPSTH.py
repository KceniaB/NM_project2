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
#  1. PSTH plot (Peri-feedback activity)
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
# ==============================================================
#  2. PSTH plot - function and different variations of the plot
#
# PSTH for stimOn and feedback events, split by correct and incorrect
#
# variables to change: 
#   EVENT = "stimOnTrigger_times" or "feedback_times"
#   time_window = (-1, 2)
#   ylim = (-2, 3)
#   save_path = ... 
# ============================================================== 
import numpy as np
import matplotlib.pyplot as plt

def plot_psth(df_nph, df_trials, subject, region, date, i=None,
              event="feedback_times", time_window=(-1, 2),
              ylim=(-2, 3), save_path=None, show=True):
    """
    Plot peri-event photometry PSTH aligned to behavioral events.

    Parameters
    ----------
    df_nph :
        Photometry dataframe containing columns ['times', 'zdFF'].
    df_trials : 
        Behavioral dataframe containing event timestamps and 'feedbackType'.
    subject
    region 
    date
        Session date (YYYY-MM-DD).
    i 
    event : 
        Event column to align to ('stimOnTrigger_times' or 'feedback_times').
    time_window : default (-1, 2)
        Time window (seconds before and after event).
    ylim : default (-2, 3)
    save_path
    show
    ----------
    """

    try:
        # --- Parameters
        SAMPLING_RATE = int(1 / np.mean(np.diff(df_nph.times)))
        t = df_nph["times"].values
        calcium = df_nph["zdFF"].values
        t_events = df_trials[event].dropna().values
        n_trials = len(t_events)
        samples_window = np.arange(time_window[0] * SAMPLING_RATE,
                                   time_window[1] * SAMPLING_RATE)
        psth_idx = np.tile(samples_window[:, np.newaxis], (1, n_trials))
        event_idx = np.searchsorted(t, t_events)
        psth_idx += event_idx

        # Mask invalid indices
        psth_idx = psth_idx[(psth_idx >= 0) & (psth_idx < len(t))].reshape(-1, n_trials)

        # Compute PSTH
        photometry_feedback = calcium[psth_idx]

        # Build time axis
        n_timepoints = photometry_feedback.shape[0]
        time_axis = np.linspace(time_window[0], time_window[1], n_timepoints)

        # --- Plot trial-wise and mean traces
        plt.figure(figsize=(8, 6), dpi=300)
        mask_correct = df_trials.feedbackType.values == 1
        mask_incorrect = ~mask_correct

        # All trials
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
        plt.title(f"{i if i is not None else ''} PSTH peri-{event} — {subject} {region} {date}")
        plt.xlabel("Time (s)")
        plt.ylabel("ΔF/F (z-scored)")
        plt.legend(frameon=False)
        plt.ylim(ylim)
        plt.tight_layout()

        # Clean style
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Save / Show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🖼️ Saved plot -> {save_path}")

            # Also save a PDF version automatically
            base, ext = os.path.splitext(save_path)
            pdf_path = base + ".pdf"
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            print(f"📄 Saved plot -> {pdf_path}")


        if show:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()

        return time_axis, photometry_feedback

    except Exception as e:
        print(f"⚠️ Could not plot PSTH for {subject} {date} {region}: {e}")
        return None, None
    
for EVENT in ["stimOnTrigger_times", "feedback_times"]: 
    plot_psth(df_nph, df_trials, subject, region, date, i=i,
                event=EVENT, time_window=(-1, 2),
                ylim=(-2, 3), 
                save_path="/home/kceniabougrova/Documents/2025_10_31_POSTER_SfN/01_psth_DA_example/psth_example_DA_1session.png", 
                show=True)


# %%
import numpy as np
import matplotlib.pyplot as plt

def compute_psth(df_nph, df_trials, event, time_window=(-1, 2)):
    """Return peri-event aligned zdFF and time axis."""
    SAMPLING_RATE = int(1 / np.mean(np.diff(df_nph.times)))
    t = df_nph["times"].values
    calcium = df_nph["zdFF"].values
    t_events = df_trials[event].dropna().values

    n_trials = len(t_events)
    samples_window = np.arange(time_window[0]*SAMPLING_RATE, time_window[1]*SAMPLING_RATE)
    psth_idx = np.tile(samples_window[:, None], (1, n_trials))
    event_idx = np.searchsorted(t, t_events)
    psth_idx += event_idx
    psth_idx = psth_idx[(psth_idx >= 0) & (psth_idx < len(t))].reshape(-1, n_trials)

    photometry = calcium[psth_idx]
    time_axis = np.linspace(time_window[0], time_window[1], photometry.shape[0])
    return time_axis, photometry


def plot_psth_grid(df_nph, df_trials, subject, region, date,
                   events=None, time_window=(-1, 2), ylim=(-2,3)):
    """
    Create 5x5 grid:
    Rows = trial groupings, Cols = events
    """
    if events is None:
        events = ["intervals_0", "stimOnTrigger_times", "firstMovement_times", "feedback_times", "intervals_1"]

    fig, axes = plt.subplots(5, 5, figsize=(20, 18), dpi=300, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    mask_correct = df_trials.feedbackType.values == 1
    mask_incorrect = ~mask_correct
    contrasts = np.sort(df_trials.signed_contrast.unique()) if "signed_contrast" in df_trials.columns else []
    probs = np.sort(df_trials.probabilityLeft.unique()) if "probabilityLeft" in df_trials.columns else []

    for col, event in enumerate(events):
        time_axis, psth = compute_psth(df_nph, df_trials, event, time_window)

        # 1️⃣ Row 1: all correct vs incorrect
        ax = axes[0, col]
        mean_c = np.nanmean(psth[:, mask_correct], axis=1)
        mean_i = np.nanmean(psth[:, mask_incorrect], axis=1)
        ax.plot(time_axis, mean_c, color="#0077b6", lw=2.5)
        ax.plot(time_axis, mean_i, color="red", lw=2.5)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_title(event.replace("_times", ""), fontsize=10)

        # 2️⃣ Row 2: correct by contrast
        ax = axes[1, col]
        if len(contrasts) > 0:
            for c in contrasts:
                m = mask_correct & (df_trials.signed_contrast == c)
                if m.sum() < 5: continue
                ax.plot(time_axis, np.nanmean(psth[:, m], axis=1), label=f"{c}")
            ax.legend(frameon=False, fontsize=6)

        # 3️⃣ Row 3: incorrect by contrast
        ax = axes[2, col]
        if len(contrasts) > 0:
            for c in contrasts:
                m = mask_incorrect & (df_trials.signed_contrast == c)
                if m.sum() < 5: continue
                ax.plot(time_axis, np.nanmean(psth[:, m], axis=1), label=f"{c}")
            ax.legend(frameon=False, fontsize=6)

        # 4️⃣ Row 4: correct by probabilityLeft
        ax = axes[3, col]
        if len(probs) > 0:
            for p in probs:
                m = mask_correct & (df_trials.probabilityLeft == p)
                if m.sum() < 5: continue
                ax.plot(time_axis, np.nanmean(psth[:, m], axis=1), label=f"pL={p}")
            ax.legend(frameon=False, fontsize=6)

        # 5️⃣ Row 5: incorrect by probabilityLeft
        ax = axes[4, col]
        if len(probs) > 0:
            for p in probs:
                m = mask_incorrect & (df_trials.probabilityLeft == p)
                if m.sum() < 5: continue
                ax.plot(time_axis, np.nanmean(psth[:, m], axis=1), label=f"pL={p}")
            ax.legend(frameon=False, fontsize=6)

    # Common labels and cosmetics
    for ax in axes.flat:
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(ylim)

    for r, label in enumerate([
        "Correct vs Incorrect",
        "Correct by Contrast",
        "Incorrect by Contrast",
        "Correct by pLeft",
        "Incorrect by pLeft"
    ]):
        axes[r,0].set_ylabel(label, fontsize=9)

    fig.suptitle(f"{subject} – {region} – {date} PSTH Grid", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# %%
plot_psth_grid(df_nph, df_trials,
               subject="ZFM-04019",
               region="DR",
               date="2023-07-15")

# %%
# ==============================================================
# 3. Plot a grid 5x5 
        # Columns (5):
            # intervals_0
            # stimOnTrigger_times
            # firstMovement_times
            # feedback_times
            # intervals_1

        # Rows (5):
            # Correct vs Incorrect (all trials)
            # Correct split by all contrasts
            # Incorrect split by all contrasts
            # Correct split by probabilityLeft
            # Incorrect split by probabilityLeft
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# =========================================================
# COMPUTE PSTH FUNCTION
# =========================================================
def compute_psth(df_nph, df_trials, event, time_window=(-1, 2)):
    """Return peri-event aligned zdFF, time axis, and valid trial indices."""
    if event not in df_trials.columns:
        print(f"⚠️ Event '{event}' not in df_trials.columns — skipped.")
        return None, None, None

    t_events = df_trials[event].dropna().values
    if len(t_events) == 0:
        print(f"⚠️ No valid timestamps for event '{event}' — skipped.")
        return None, None, None

    SAMPLING_RATE = int(1 / np.mean(np.diff(df_nph.times)))
    t = df_nph["times"].values
    calcium = df_nph["zdFF"].values

    n_trials = len(t_events)
    samples_window = np.arange(time_window[0]*SAMPLING_RATE, time_window[1]*SAMPLING_RATE)
    psth_idx = np.tile(samples_window[:, None], (1, n_trials))
    event_idx = np.searchsorted(t, t_events)
    psth_idx += event_idx

    # Mask invalid trials (events too close to signal edges)
    valid_trials = (psth_idx.min(axis=0) >= 0) & (psth_idx.max(axis=0) < len(t))
    if not np.all(valid_trials):
        bad_idx = np.where(~valid_trials)[0]
        for b in bad_idx:
            print(f"⚠️ Skipped trial #{b} for event '{event}' (out of bounds)")
        psth_idx = psth_idx[:, valid_trials]

    if psth_idx.size == 0:
        print(f"⚠️ Empty PSTH for event '{event}' — skipped.")
        return None, None, None

    photometry = calcium[psth_idx]
    time_axis = np.linspace(time_window[0], time_window[1], photometry.shape[0])

    kept_trials = np.where(df_trials[event].notna())[0][valid_trials]
    return time_axis, photometry, kept_trials


# =========================================================
# 5×5 GRID PLOTTING FUNCTION
# =========================================================
def plot_psth_grid(df_nph, df_trials, subject, region, date,
                   events=None, time_window=(-1, 2), ylim=(-2,3)):
    """
    Create 5x5 PSTH grid:
    Rows = trial groupings (Correct/Incorrect, contrasts, probabilities)
    Cols = behavioral events
    """
    if events is None:
        events = ["intervals_0", "stimOnTrigger_times", "firstMovement_times", "feedback_times", "intervals_1"]

    fig, axes = plt.subplots(5, 5, figsize=(20, 18), dpi=300, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    # Extract unique values if available
    contrasts = np.sort(df_trials.allContrasts.unique()) if "allContrasts" in df_trials.columns else []
    probs = np.sort(df_trials.probabilityLeft.unique()) if "probabilityLeft" in df_trials.columns else []

    # Generate gradient colormap for contrasts
    if len(contrasts) > 0:
        cmap = cm.get_cmap("inferno_r", len(contrasts))
        contrast_colors = [colors.to_hex(cmap(i)) for i in range(len(contrasts))]
    else:
        contrast_colors = []

    for col, event in enumerate(events):
        time_axis, psth, kept = compute_psth(df_nph, df_trials, event, time_window)
        if psth is None:
            for row in range(5):
                axes[row, col].set_axis_off()
            continue

        # Adjust trial-level masks for surviving trials only
        mask_correct = df_trials.feedbackType.values[kept] == 1
        mask_incorrect = ~mask_correct

        # # 1️⃣ Row 1: all correct vs incorrect
        # ax = axes[0, col]
        # try:
        #     mean_c = np.nanmean(psth[:, mask_correct], axis=1)
        #     mean_i = np.nanmean(psth[:, mask_incorrect], axis=1)
        #     ax.plot(time_axis, mean_c, color="#0077b6", lw=2.5)
        #     ax.plot(time_axis, mean_i, color="red", lw=2.5)
        # except Exception as e:
        #     print(f"⚠️ Could not plot Correct/Incorrect for event '{event}': {e}")
        #     ax.set_axis_off()
        # ax.axvline(0, color="black", ls="--", lw=1)
        # ax.set_title(event.replace("_times", ""), fontsize=10)

        # 1️⃣ Row 1: all correct vs incorrect
        ax = axes[0, col]
        try:
            mean_c = np.nanmean(psth[:, mask_correct], axis=1)
            sem_c = np.nanstd(psth[:, mask_correct], axis=1) / np.sqrt(mask_correct.sum())
            mean_i = np.nanmean(psth[:, mask_incorrect], axis=1)
            sem_i = np.nanstd(psth[:, mask_incorrect], axis=1) / np.sqrt(mask_incorrect.sum())

            ax.plot(time_axis, mean_c, color="#0077b6", lw=2.5)
            ax.fill_between(time_axis, mean_c - sem_c, mean_c + sem_c, color="#0077b6", alpha=0.3)
            ax.plot(time_axis, mean_i, color="red", lw=2.5)
            ax.fill_between(time_axis, mean_i - sem_i, mean_i + sem_i, color="red", alpha=0.3)

        except Exception as e:
            print(f"⚠️ Could not plot Correct/Incorrect for event '{event}': {e}")
            ax.set_axis_off()
        ax.legend(frameon=False, fontsize=6, loc="upper right")
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_title(event.replace("_times", ""), fontsize=10)


        # # 2️⃣ Row 2: correct by contrast
        # ax = axes[1, col]
        # if len(contrasts) > 0:
        #     for c_idx, c in enumerate(contrasts):
        #         m = mask_correct & (df_trials.allContrasts.values[kept] == c)
        #         if m.sum() < 5: 
        #             continue
        #         ax.plot(time_axis, np.nanmean(psth[:, m], axis=1), color=contrast_colors[c_idx], lw=2.0, label=f"{c}")
        #     if ax.has_data():
        #         ax.legend(frameon=False, fontsize=6, title="Contrast")
        #     else:
        #         print(f"⚠️ No valid Correct-by-Contrast data for event '{event}'")

        # 2️⃣ Row 2: correct by contrast
        ax = axes[1, col]
        if len(contrasts) > 0:
            for c_idx, c in enumerate(contrasts):
                m = mask_correct & (df_trials.allContrasts.values[kept] == c)
                if m.sum() < 5: 
                    continue
                y = np.nanmean(psth[:, m], axis=1)
                sem = np.nanstd(psth[:, m], axis=1) / np.sqrt(m.sum())
                ax.plot(time_axis, y, color=contrast_colors[c_idx], lw=2.0, label=f"{c}")
                ax.fill_between(time_axis, y - sem, y + sem, color=contrast_colors[c_idx], alpha=0.3)
            ax.axvline(0, color="black", ls="--", lw=1)
            if ax.has_data():
                ax.legend(frameon=False, fontsize=6, title="Contrast")


        # # 3️⃣ Row 3: incorrect by contrast
        # ax = axes[2, col]
        # if len(contrasts) > 0:
        #     for c_idx, c in enumerate(contrasts):
        #         m = mask_incorrect & (df_trials.allContrasts.values[kept] == c)
        #         if m.sum() < 5:
        #             continue
        #         ax.plot(time_axis, np.nanmean(psth[:, m], axis=1), color=contrast_colors[c_idx], lw=2.0, label=f"{c}")
        #     if ax.has_data():
        #         ax.legend(frameon=False, fontsize=6, title="Contrast")
        #     else:
        #         print(f"⚠️ No valid Incorrect-by-Contrast data for event '{event}'") 

        # 3️⃣ Row 3: incorrect by contrast
        ax = axes[2, col]
        if len(contrasts) > 0:
            for c_idx, c in enumerate(contrasts):
                m = mask_incorrect & (df_trials.allContrasts.values[kept] == c)
                if m.sum() < 5:
                    continue
                y = np.nanmean(psth[:, m], axis=1)
                sem = np.nanstd(psth[:, m], axis=1) / np.sqrt(m.sum())
                ax.plot(time_axis, y, color=contrast_colors[c_idx], lw=2.0, label=f"{c}")
                ax.fill_between(time_axis, y - sem, y + sem, color=contrast_colors[c_idx], alpha=0.3)
            ax.axvline(0, color="black", ls="--", lw=1)
            if ax.has_data():
                ax.legend(frameon=False, fontsize=6, title="Contrast")


        # # 4️⃣ Row 4: correct by probabilityLeft
        # ax = axes[3, col]
        # if len(probs) > 0:
        #     for p in probs:
        #         m = mask_correct & (df_trials.probabilityLeft.values[kept] == p)
        #         if m.sum() < 5: continue
        #         ax.plot(time_axis, np.nanmean(psth[:, m], axis=1), lw=2.0, label=f"pL={p}")
        #     if ax.has_data():
        #         ax.legend(frameon=False, fontsize=6)
        #     else:
        #         print(f"⚠️ No valid Correct-by-pLeft data for event '{event}'") 

        # 4️⃣ Row 4: correct by probabilityLeft
        ax = axes[3, col]
        if len(probs) > 0:
            for p in probs:
                m = mask_correct & (df_trials.probabilityLeft.values[kept] == p)
                if m.sum() < 5: continue
                y = np.nanmean(psth[:, m], axis=1)
                sem = np.nanstd(psth[:, m], axis=1) / np.sqrt(m.sum())
                ax.plot(time_axis, y, lw=2.0, label=f"pL={p}")
                ax.fill_between(time_axis, y - sem, y + sem, alpha=0.3)
            ax.axvline(0, color="black", ls="--", lw=1)
            if ax.has_data():
                ax.legend(frameon=False, fontsize=6)

        # # 5️⃣ Row 5: incorrect by probabilityLeft
        # ax = axes[4, col]
        # if len(probs) > 0:
        #     for p in probs:
        #         m = mask_incorrect & (df_trials.probabilityLeft.values[kept] == p)
        #         if m.sum() < 5: continue
        #         ax.plot(time_axis, np.nanmean(psth[:, m], axis=1), lw=2.0, label=f"pL={p}")
        #     if ax.has_data():
        #         ax.legend(frameon=False, fontsize=6)
        #     else:
        #         print(f"⚠️ No valid Incorrect-by-pLeft data for event '{event}'") 

        # 5️⃣ Row 5: incorrect by probabilityLeft
        ax = axes[4, col]
        if len(probs) > 0:
            for p in probs:
                m = mask_incorrect & (df_trials.probabilityLeft.values[kept] == p)
                if m.sum() < 5: continue
                y = np.nanmean(psth[:, m], axis=1)
                sem = np.nanstd(psth[:, m], axis=1) / np.sqrt(m.sum())
                ax.plot(time_axis, y, lw=2.0, label=f"pL={p}")
                ax.fill_between(time_axis, y - sem, y + sem, alpha=0.3)
            ax.axvline(0, color="black", ls="--", lw=1)
            if ax.has_data():
                ax.legend(frameon=False, fontsize=6)


    # --- Force legend display for first row (Correct vs Incorrect)
    for col in range(len(events)):
        ax = axes[0, col]
        lines = ax.get_lines()
        if len(lines) > 0:
            ax.legend(
                handles=lines[:2],  
                labels=["Correct", "Incorrect"],
                frameon=False,
                fontsize=6,
                loc="upper right"
            )

    # ---- plot format ----
    for ax in axes.flat:
        if not ax.has_data():
            ax.set_axis_off()
            continue
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(ylim)

    # Row labels
    for r, label in enumerate([
        "Correct vs Incorrect",
        "Correct by Contrast",
        "Incorrect by Contrast",
        "Correct by pLeft",
        "Incorrect by pLeft"
    ]):
        axes[r,0].set_ylabel(label, fontsize=9)

    fig.suptitle(f"{subject} – {region} – {date} PSTH Grid", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show() 


plot_psth_grid(
    df_nph, df_trials,
    subject="ZFM-04019",
    region=region,
    date=date
)



# %%
