#%%
"""
KB
16_V3_redoing.py_OCT062025_predictNMshape.py
30-October-2025

DONE

Runs through the selected rows from the csv file of all the photometry sessions
Check the plots and manually label the ones with a good signal 
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


old_prefix = '/mnt/h0/kb/'
new_prefix = '/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/'



#%% 
# =========================================================
# CONFIG
# =========================================================
PLOT = True

processed_sessions = []
error_sessions_processing = []

# =========================================================
# LOOP THROUGH FILTERED SESSIONS
# =========================================================
start_idx = 0 #1566 #1209 #1159 #1138 #1089 #1073 #1058 #987 #921 #608 #523 #345 #149
end_idx = 155
for idx, row in sessions_list.iloc[:end_idx].iterrows():
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




