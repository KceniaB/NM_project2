#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import seaborn as sns

# path_to_data = '/mnt/h0/kb/Feb2025/data_behav_psth/Good_sessions/'
path_to_data = '/home/kceniabougrova/Documents/NM_project_fromIBLserver/Good_sessions'
EVENTS = ['feedback_times']

groups = {
    "DA": ["ZFM-03447", "ZFM-04026", "ZFM-04022_R", "ZFM-03448", "ZFM-04019", "ZFM-04022_L"],
    # "VTA": ["ZFM-03447", "ZFM-04026", "ZFM-04022_R"],
    # "SNc": ["ZFM-03448", "ZFM-04019", "ZFM-04022_L"],
    "5-HT": ["ZFM-03059", "ZFM-03065", "ZFM-03061", "ZFM-03062", "ZFM-04392_DRN", "ZFM-05235", "ZFM-05236", "ZFM-05245", "ZFM-05248"],
    # "DR": ["ZFM-03059", "ZFM-03065", "ZFM-03061", "ZFM-04392_DRN", "ZFM-05235", "ZFM-05236", "ZFM-05245", "ZFM-05248"],
    # "MR": ["ZFM-03062"],
    "NE": ["ZFM-04533", "ZFM-04534_LC", "ZFM-06268", "ZFM-06271", "ZFM-06272_L", "ZFM-06272_R", "ZFM-06171", "ZFM-06275_L", "ZFM-06275_R"],
    # "LC_L": ["ZFM-04533", "ZFM-04534_LC", "ZFM-06268", "ZFM-06271", "ZFM-06272_L", "ZFM-06171", "ZFM-06275_L"],
    # "LC_R": ["ZFM-06272_R", "ZFM-06275_R"],
    # "ACh": ["ZFM-06305_L", "ZFM-06305_R", "ZFM-06948"],
    "ACh": ["ZFM-06305_L", "ZFM-06948"]
    # "SI_R": ["ZFM-06305_R"]
} 


# %%
#===========================================================================
#                            Funtions
#===========================================================================
def interpolate_to_common_time(psth, current_rate, target_rate, PERIEVENT_WINDOW):
    """Interpolate psth data to match the target sampling rate."""
    current_time = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], psth.shape[0])
    target_time = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], int((PERIEVENT_WINDOW[1] - PERIEVENT_WINDOW[0]) * target_rate))
    interpolated_psth = np.array([np.interp(target_time, current_time, psth[:, i]) for i in range(psth.shape[1])]).T
    return interpolated_psth

def plot_heatmap_psth_group(df_trials, psth_combined, EVENT="feedback_times", group_name='', PERIEVENT_WINDOW=[-1, 2], SAMPLING_RATE=30):
    """Plot side-by-side line plots for correct and incorrect trials with enhanced aesthetics."""
    
    # Extract correct and incorrect trials
    # psth_good = psth_combined[:, df_trials.feedbackType == 1]
    # psth_error = psth_combined[:, df_trials.feedbackType == -1]
    psth_good = psth_combined[:, df_trials.feedbackType.values == 1]
    psth_error = psth_combined[:, df_trials.feedbackType.values == -1]


    # Compute mean and SEM
    psth_good_avg = psth_good.mean(axis=1)
    sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
    psth_error_avg = psth_error.mean(axis=1)
    sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

    # Time axis
    time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], psth_combined.shape[0]) 

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300, sharey=True)

    # Define colors
    correct_color = '#2f9c95'
    incorrect_color = '#d62828'
    shadow_color_correct = '#577590'
    shadow_color_incorrect = '#903749'

    # Correct trials average plot
    axes[0].plot(time_axis, psth_good_avg, linewidth=2.5, color=correct_color, label="Correct")
    axes[0].fill_between(time_axis, psth_good_avg - sem_good, psth_good_avg + sem_good, color=shadow_color_correct, alpha=0.15)
    axes[0].axvline(x=0, color="black", linestyle="--", linewidth=2, label="Event Onset")
    axes[0].set_xlabel(f"Time since {EVENT} (s)", fontsize=20)
    axes[0].set_ylabel("Calcium Signal", fontsize=20)
    axes[0].set_title(f"Correct Trials", fontsize=20, pad=15)
    
    # Incorrect trials average plot
    axes[1].plot(time_axis, psth_error_avg, linewidth=2.5, color=incorrect_color, label="Incorrect")
    axes[1].fill_between(time_axis, psth_error_avg - sem_error, psth_error_avg + sem_error, color=shadow_color_incorrect, alpha=0.15)
    axes[1].axvline(x=0, color="black", linestyle="--", linewidth=2, label="Event Onset")
    axes[1].set_xlabel(f"Time since {EVENT} (s)", fontsize=20)
    axes[1].set_title(f"Incorrect Trials", fontsize=20, pad=15)


    # Remove top and right spines
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=16)

    # Create custom legend patches for shaded error regions
    shadow_legend_correct = mpatches.Patch(color=shadow_color_correct, alpha=0.15, label="SEM (Correct)")
    shadow_legend_incorrect = mpatches.Patch(color=shadow_color_incorrect, alpha=0.15, label="SEM (Incorrect)")

    # Create legend and place it outside
    legend = axes[1].legend(
        handles=[shadow_legend_correct, shadow_legend_incorrect] + axes[1].get_legend_handles_labels()[0], 
        fontsize=16, frameon=False, loc="upper left", bbox_to_anchor=(1, 1), handlelength=1.8, handleheight=1
    )

    plt.suptitle(f'{group_name}_{EVENT}', fontsize=14)
    plt.tight_layout()
    plt.show() 
#     # Save figures
    # save_path_png = f'/mnt/h0/kb/Feb2025/plots/Chapter4/Fig02_perNM_{df_trials.mouse.iloc[0]}_{df_trials.date.iloc[0]}_{group_name}_correct.png'
    # save_path_pdf = f'/mnt/h0/kb/Feb2025/plots/Chapter4/Fig02_perNM_{df_trials.mouse.iloc[0]}_{df_trials.date.iloc[0]}_{group_name}_correct.pdf'
    # save_path_png = f'/mnt/h0/kb/Feb2025/plots/Chapter4/Fig02_perNM_{df_trials.mouse.iloc[0]}_{df_trials.date.iloc[0]}_{group_name}_incorrect.png'
    # save_path_pdf = f'/mnt/h0/kb/Feb2025/plots/Chapter4/Fig02_perNM_{df_trials.mouse.iloc[0]}_{df_trials.date.iloc[0]}_{group_name}_incorrect.pdf'

    # plt.savefig(save_path_png, dpi=300)
    # plt.savefig(save_path_pdf, dpi=300)
    plt.show()



def pad_to_match_length(arr, target_length):
    """Pad or trim an array to match the target length along the time axis."""
    current_length = arr.shape[0]
    if current_length == target_length:
        return arr
    elif current_length < target_length:
        padding = target_length - current_length
        return np.pad(arr, ((0, padding), (0, 0)), mode='constant')
    else:
        return arr[:target_length, :]

def load_group_data(group_mice, event, path_to_data, PERIEVENT_WINDOW):
    """Load and pad/interpolate psth and df_trials for a group of mice to a common length."""
    combined_psth = []
    combined_trials = []
    target_rate = 30
    max_time_length = 0

    for mouse in group_mice:
        try:
            psth_path_30 = os.path.join(path_to_data, f'psth_{mouse}_{event}_allsessions_30Hz.npy')
            psth_path_15 = os.path.join(path_to_data, f'psth_{mouse}_{event}_allsessions_15Hz.npy')
            df_trials_path_30 = os.path.join(path_to_data, f'df_trials_{mouse}_{event}_allsessions_30Hz.pqt')
            df_trials_path_15 = os.path.join(path_to_data, f'df_trials_{mouse}_{event}_allsessions_15Hz.pqt')
            
            if os.path.exists(psth_path_30):
                psth = np.load(psth_path_30)
                df_trials = pd.read_parquet(df_trials_path_30)
            elif os.path.exists(psth_path_15):
                psth = np.load(psth_path_15)
                psth = interpolate_to_common_time(psth, current_rate=15, target_rate=30, PERIEVENT_WINDOW=PERIEVENT_WINDOW)
                df_trials = pd.read_parquet(df_trials_path_15)
            else:
                continue

            max_time_length = max(max_time_length, psth.shape[0])
            combined_psth.append(psth)
            combined_trials.append(df_trials)
        
        except Exception as e:
            print(f"Error loading data for {mouse} | Event: {event}: {e}")
    
    if combined_psth and combined_trials:
        # Pad all PSTH arrays to match the longest one
        combined_psth = [pad_to_match_length(psth, max_time_length) for psth in combined_psth]
        psth_combined = np.concatenate(combined_psth, axis=1)
        df_trials_combined = pd.concat(combined_trials, ignore_index=True)
        return psth_combined, df_trials_combined
    else:
        return None, None 
    
#%%
#===========================================================================
EVENTS = ['stimOnTrigger_times', 'feedback_times']
for group_name, group_mice in groups.items():
    for event in EVENTS:
        print(f"Processing group {group_name} | Event: {event}")
        psth_combined, df_trials_combined = load_group_data(group_mice, event, path_to_data, PERIEVENT_WINDOW=[-1, 2])
        
        if psth_combined is not None:
            plot_heatmap_psth_group(
                df_trials=df_trials_combined,
                psth_combined=psth_combined,
                EVENT=event,
                group_name=group_name,
                PERIEVENT_WINDOW=[-1, 2]
            )
        else:
            print(f"No data found for group {group_name} | Event: {event}.") 

#%%
#===========================================================================
def plot_correct_vs_incorrect_by_NM(df_trials_dict, psth_dict, EVENT="feedback_times", PERIEVENT_WINDOW=[-1, 2], SAMPLING_RATE=30):
    """Plot side-by-side line plots for correct and incorrect trials across NM groups."""

    # Define colors for each NM
    color_palette = {
        "DA": "#FF3B25",   # Red
        "5-HT": "#9F4DF1", # Purple
        "NE": "#2198ED",   # Blue
        "ACh": "#45B437"   # Green
    }
    
    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)

    time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], psth_dict["DA"].shape[0]) 

    for nm, color in color_palette.items():
        if nm in psth_dict and psth_dict[nm] is not None:
            # Extract correct and incorrect trials
            psth_correct = psth_dict[nm][:, df_trials_dict[nm].feedbackType.values == 1]
            psth_incorrect = psth_dict[nm][:, df_trials_dict[nm].feedbackType.values == -1]

            # Compute means and SEM
            psth_correct_avg = psth_correct.mean(axis=1)
            sem_correct = psth_correct.std(axis=1) / np.sqrt(psth_correct.shape[1]) if psth_correct.shape[1] > 0 else np.zeros_like(psth_correct_avg)

            psth_incorrect_avg = psth_incorrect.mean(axis=1)
            sem_incorrect = psth_incorrect.std(axis=1) / np.sqrt(psth_incorrect.shape[1]) if psth_incorrect.shape[1] > 0 else np.zeros_like(psth_incorrect_avg)

            # Plot correct trials (Left)
            axes[0].plot(time_axis, psth_correct_avg, linewidth=2.5, color=color, label=f"{nm}")
            axes[0].fill_between(time_axis, psth_correct_avg - sem_correct, psth_correct_avg + sem_correct, color=color, alpha=0.15)

            # Plot incorrect trials (Right)
            axes[1].plot(time_axis, psth_incorrect_avg, linewidth=2.5, color=color, label=f"{nm}")
            axes[1].fill_between(time_axis, psth_incorrect_avg - sem_incorrect, psth_incorrect_avg + sem_incorrect, color=color, alpha=0.15)

    # Formatting and aesthetics
    for ax in axes:
        ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Event Onset")
        # ax.set_xlabel(f"Time since {EVENT} (s)", fontsize=20)
        # ax.set_xlabel(f"Time since stimulus onset (s)", fontsize=20)
        ax.set_xlabel(f"Time since feedback (s)", fontsize=20)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=16)

    axes[0].set_ylabel("Calcium Signal", fontsize=20)
    axes[0].set_title("Correct Trials", fontsize=20, pad=30)
    axes[1].set_title("Incorrect Trials", fontsize=20, pad=30)

    # Create legend
    legend_handles = [mpatches.Patch(color=color, label=nm) for nm, color in color_palette.items()]
    legend = axes[1].legend(handles=legend_handles, fontsize=16, frameon=False, loc="upper left", bbox_to_anchor=(1, 1), handlelength=1.8, handleheight=0.2)

    plt.suptitle(f"Correct and incorrect trials across neuromodulators", fontsize=22, y=1.05)
    plt.tight_layout()

df_trials_dict = {}
psth_dict = {} 

#%%
#===========================================================================

for group_name, group_mice in groups.items():
    # psth_combined, df_trials_combined = load_group_data(group_mice, "stimOnTrigger_times", path_to_data, PERIEVENT_WINDOW=[-1, 2])
    psth_combined, df_trials_combined = load_group_data(group_mice, "feedback_times", path_to_data, PERIEVENT_WINDOW=[-1, 2])
    
    if psth_combined is not None:
        df_trials_dict[group_name] = df_trials_combined
        psth_dict[group_name] = psth_combined
    else:
        print(f"No data found for {group_name}")

# Call the new plotting function
# plot_correct_vs_incorrect_by_NM(df_trials_dict, psth_dict, EVENT="stimOnTrigger_times")
plot_correct_vs_incorrect_by_NM(df_trials_dict, psth_dict, EVENT="feedback_times") 

# %%
#===========================================================================
groups = {
    "DA": ["ZFM-03447", "ZFM-04026", "ZFM-04022_R", "ZFM-03448", "ZFM-04019", "ZFM-04022_L"]
}

EVENTS = ['stimOnTrigger_times', 'feedback_times']
for group_name, group_mice in groups.items():
    for event in EVENTS:
        print(f"Processing group {group_name} | Event: {event}")
        psth_combined, df_trials_combined = load_group_data(group_mice, event, path_to_data, PERIEVENT_WINDOW=[-1, 2])


if psth_combined.shape[0] == 91:
    psth_combined = psth_combined.T  # Now shape = (10628, 91)

# Compute mean across trials (axis 0), resulting in 91 values
mean_psth = np.mean(psth_combined, axis=0)

# Plotting
plt.plot(mean_psth)
plt.xlabel("Time Bin")
plt.ylabel("Mean Response")
plt.title("Average PSTH")
plt.grid(True)
plt.axvline(x=30, color='r', linestyle='--', label='Event Onset')
plt.show()





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Time vector from -1 to 2 seconds, 91 bins
time_vector = np.linspace(-1, 2, 91)

# 2. Define ranges
baseline_range = (-0.2, 0)
response_range = (0, 0.5)

# 3. Get index positions for baseline and response
baseline_idx = np.where((time_vector >= baseline_range[0]) & (time_vector < baseline_range[1]))[0]
response_idx = np.where((time_vector >= response_range[0]) & (time_vector < response_range[1]))[0]

# 4. Check and transpose if needed
if psth_combined.shape[0] == 91:
    psth_combined = psth_combined.T  # shape becomes (35690, 91)

# 5. Average PSTH signal over each time window (per row/trial)
baseline_avg = np.mean(psth_combined[:, baseline_idx], axis=1)  
response_avg = np.mean(psth_combined[:, response_idx], axis=1)  

# 6. Compute raw difference and normalized difference
raw_diff = response_avg - baseline_avg
epsilon = 1e-6  # to avoid division by zero
normalized_diff = raw_diff / (baseline_avg + epsilon)  # shape: (35690,)


# Prepare data in long format
df_box = pd.DataFrame({
    "Baseline": baseline_avg,
    "Response": response_avg, 
    "Raw_diff": raw_diff
})

# Melt the dataframe to long format for seaborn
df_melted = df_box.melt(var_name="Time Window", value_name="Avg ΔF/F")

# Plot boxplot
plt.figure(figsize=(6, 5))
sns.boxplot(data=df_melted, x="Time Window", y="Avg ΔF/F", palette=["skyblue", "lightgreen", "coral"])
sns.stripplot(data=df_melted, x="Time Window", y="Avg ΔF/F", palette=["gray", "gray", "gray"], size=2, alpha=0.01, jitter=True)
plt.title("Boxplot of PSTH Averages\nBaseline vs Response")
plt.tight_layout()
plt.ylim(-0.02,0.025)
plt.show()

# %%
#===========================================================================
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Time vector
time_vector = np.linspace(-1, 2, 91)

# Step 2: Get unique sorted contrast levels
df_trials_combined["allContrasts"] = pd.to_numeric(df_trials_combined["allContrasts"], errors="coerce")
contrast_levels = sorted(df_trials_combined["allContrasts"].dropna().unique())

# Step 3: Plot
plt.figure(figsize=(10, 6))

colors = plt.cm.viridis_r(np.linspace(0, 1, len(contrast_levels)))  # color per contrast

for i, contrast in enumerate(contrast_levels):
    # Create mask for correct trials at this contrast
    mask = (df_trials_combined["feedbackType"] == 1) & (df_trials_combined["allContrasts"] == contrast)
    
    if mask.sum() == 0:
        continue  # skip if no trials

    # Extract and compute mean & SEM
    psth_subset = psth_combined[mask.values, :]  # shape: (n_trials, 91)
    mean_psth = np.mean(psth_subset, axis=0)
    sem_psth = np.std(psth_subset, axis=0) / np.sqrt(psth_subset.shape[0])

    # Plot line and shaded area
    plt.plot(time_vector, mean_psth, label=f"{contrast}", color=colors[i])
    plt.fill_between(time_vector, mean_psth - sem_psth, mean_psth + sem_psth, color=colors[i], alpha=0.2)

# Final formatting
plt.axvline(0, color="black", linestyle="--", label="Event")
plt.title("PSTH by Contrast (Correct Trials Only)")
plt.xlabel("Time (s) from Event")
plt.ylabel("ΔF/F")
plt.legend(title="Contrast", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
#===========================================================================
""" 27May2025 - working """
import numpy as np

# 0. choose the event 
groups = {
    "DA": ["ZFM-03447", "ZFM-04026", "ZFM-04022_R", "ZFM-03448", "ZFM-04019", "ZFM-04022_L"]
}

EVENTS = ['stimOnTrigger_times'] 
# EVENTS = ['stimOnTrigger_times', 'feedback_times'] 
for group_name, group_mice in groups.items():
    for event in EVENTS:
        print(f"Processing group {group_name} | Event: {event}")
        psth_combined, df_trials_combined = load_group_data(group_mice, event, path_to_data, PERIEVENT_WINDOW=[-1, 2])
if psth_combined.shape[0] == 91:
    psth_combined = psth_combined.T  # Now shape = (10628, 91)

        
# 1. Time vector
time_vector = np.linspace(-1, 2, 91)

# 2. Indices for baseline and response windows
baseline_idx = np.where((time_vector >= -0.2) & (time_vector < 0))[0]
response_idx = np.where((time_vector >= 0) & (time_vector < 0.5))[0]

# 3. Get unique contrasts
unique_contrasts = np.unique(df_trials_combined['allContrasts'].values)

# 4. Prepare dictionaries for correct and incorrect trials
groups = {
    "correct": df_trials_combined['feedbackType'].values == 1,
    "incorrect": df_trials_combined['feedbackType'].values == -1
}

# To store normalized and aligned PSTHs
results = {
    "correct": {
        "aligned_mean": {},
        "aligned_sem": {},
        "normalized_mean": {},
        "normalized_sem": {}
    },
    "incorrect": {
        "aligned_mean": {},
        "aligned_sem": {},
        "normalized_mean": {},
        "normalized_sem": {}
    }
}

epsilon = 1e-6

# 5. Loop over group (correct/incorrect)
for group_name, group_mask in groups.items():
    # Loop over contrasts
    for contrast in unique_contrasts:
        # Get trials for this contrast and feedbackType group
        idx = (df_trials_combined['allContrasts'].values == contrast) & group_mask
        psth_subset = psth_combined[idx]

        if psth_subset.shape[0] == 0:
            continue  # skip if no trials for this condition

        # Compute baseline mean per trial (shape: [n_trials, 1])
        baseline_mean = np.mean(psth_subset[:, baseline_idx], axis=1, keepdims=True)

        # Subtract baseline from entire PSTH trace (trial-wise)
        aligned_psth = psth_subset - baseline_mean  # shape: (n_trials, 91)

        # Normalize using trial-wise baseline
        normalized_psth = aligned_psth / (baseline_mean + epsilon)

        # Store mean and SEM across trials
        results[group_name]["aligned_mean"][contrast] = np.mean(aligned_psth, axis=0)
        results[group_name]["aligned_sem"][contrast] = np.std(aligned_psth, axis=0, ddof=1) / np.sqrt(aligned_psth.shape[0])

        results[group_name]["normalized_mean"][contrast] = np.mean(normalized_psth[:, response_idx], axis=0).squeeze()
        results[group_name]["normalized_sem"][contrast] = np.std(normalized_psth[:, response_idx], axis=0, ddof=1).squeeze() / np.sqrt(normalized_psth.shape[0])

# 6. Plot: Aligned PSTHs per contrast (correct trials)
plt.figure(figsize=(8, 6))
for contrast in sorted(results["correct"]["aligned_mean"].keys()):
    mean_trace = results["correct"]["aligned_mean"][contrast]
    sem_trace = results["correct"]["aligned_sem"][contrast]
    plt.plot(time_vector, mean_trace, label=f"Contrast {contrast:.3f}")
    plt.fill_between(time_vector, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.2)

plt.title("Correct Trials: Baseline-subtracted PSTHs by Contrast")
plt.xlabel("Time (s)")
plt.ylabel("ΔF/F (Baseline-subtracted)")
plt.axvline(0, color='k', linestyle='--', label='Event onset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#for incorrect trials
plt.figure(figsize=(8, 6))
for contrast in sorted(results["incorrect"]["aligned_mean"].keys()):
    mean_trace = results["incorrect"]["aligned_mean"][contrast]
    sem_trace = results["incorrect"]["aligned_sem"][contrast]
    plt.plot(time_vector, mean_trace, label=f"Contrast {contrast:.3f}")
    plt.fill_between(time_vector, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.2)

plt.title("Incorrect Trials: Baseline-subtracted PSTHs by Contrast")
plt.xlabel("Time (s)")
plt.ylabel("ΔF/F (Baseline-subtracted)")
plt.axvline(0, color='k', linestyle='--', label='Event onset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% 
#===========================================================================


""" working v2 """
# 0. choose the event 
groups = {
    "DA": ["ZFM-03447", "ZFM-04026", "ZFM-04022_R", "ZFM-03448", "ZFM-04019", "ZFM-04022_L"]
}

EVENTS = ['stimOnTrigger_times'] 
for group_name, group_mice in groups.items():
    for event in EVENTS:
        print(f"Processing group {group_name} | Event: {event}")
        psth_combined, df_trials_combined = load_group_data(group_mice, event, path_to_data, PERIEVENT_WINDOW=[-1, 2])

# If PSTH was transposed earlier, fix shape
if psth_combined.shape[0] == 91:
    psth_combined = psth_combined.T  # Now shape = (n_trials, 91)

# 1. Time vector and baseline indices
time_vector = np.linspace(-1, 2, 91)
baseline_idx = np.where((time_vector >= -0.2) & (time_vector < 0))[0]

# 2. Small constant to avoid divide-by-zero
epsilon = 1e-6

# 3. Masks for trial types
correct_mask = df_trials_combined['feedbackType'].values == 1
incorrect_mask = df_trials_combined['feedbackType'].values == -1

# 4. Align and normalize function (baseline-std normalization)
def align_and_normalize(psth_subset):
    # Baseline mean and std per trial
    baseline_value = np.mean(psth_subset[:, baseline_idx], axis=1, keepdims=True)
    baseline_std = np.std(psth_subset[:, baseline_idx], axis=1, keepdims=True)

    # Align: subtract baseline mean
    aligned_psth = psth_subset - baseline_value
    baseline_value_abs = abs(baseline_value)
    # Normalize: divide by baseline std
    normalized_psth = aligned_psth / (baseline_value_abs + epsilon)

    # Mean and SEM across trials
    mean_trace = np.mean(normalized_psth, axis=0)
    sem_trace = np.std(normalized_psth, axis=0, ddof=1) / np.sqrt(normalized_psth.shape[0])

    return normalized_psth, mean_trace, sem_trace

# 5. Apply to correct and incorrect trials
normalized_psth_c, mean_correct, sem_correct = align_and_normalize(psth_combined[correct_mask])
normalized_psth_inc, mean_incorrect, sem_incorrect = align_and_normalize(psth_combined[incorrect_mask])

# 6. Plot both traces
plt.figure(figsize=(8, 6))

plt.plot(time_vector, mean_correct, label='Correct', color='green')
plt.fill_between(time_vector, mean_correct - sem_correct, mean_correct + sem_correct, color='green', alpha=0.2)

plt.plot(time_vector, mean_incorrect, label='Incorrect', color='red')
plt.fill_between(time_vector, mean_incorrect - sem_incorrect, mean_incorrect + sem_incorrect, color='red', alpha=0.2)

plt.axvline(0, color='k', linestyle='--', label='Event onset')
plt.xlabel("Time (s)")
plt.ylabel("Normalized ΔF/F (baseline avg abs)")
plt.title("PSTHs normalized to baseline avg abs (Correct vs Incorrect)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
def load_group_data(group_mice, event, path_to_data, PERIEVENT_WINDOW):
    """Load and pad/interpolate psth and df_trials for a group of mice to a common length."""
    combined_psth = []
    combined_trials = []
    target_rate = 30
    max_time_length = 0

    for mouse in group_mice:
        try:
            psth_path_30 = os.path.join(path_to_data, f'psth_{mouse}_{event}_allsessions_30Hz.npy')
            psth_path_15 = os.path.join(path_to_data, f'psth_{mouse}_{event}_allsessions_15Hz.npy')
            df_trials_path_30 = os.path.join(path_to_data, f'df_trials_{mouse}_{event}_allsessions_30Hz.pqt')
            df_trials_path_15 = os.path.join(path_to_data, f'df_trials_{mouse}_{event}_allsessions_15Hz.pqt')
            
            if os.path.exists(psth_path_30):
                psth = np.load(psth_path_30)
                df_trials = pd.read_parquet(df_trials_path_30)
            elif os.path.exists(psth_path_15):
                psth = np.load(psth_path_15)
                psth = interpolate_to_common_time(psth, current_rate=15, target_rate=30, PERIEVENT_WINDOW=PERIEVENT_WINDOW)
                df_trials = pd.read_parquet(df_trials_path_15)
            else:
                continue

            # Add the 'subject_2' column to df_trials, mapping it to the current mouse
            df_trials['subject_2'] = mouse  # Add the mouse as a 'subject_2' column


            max_time_length = max(max_time_length, psth.shape[0])
            combined_psth.append(psth)
            combined_trials.append(df_trials)
        
        except Exception as e:
            print(f"Error loading data for {mouse} | Event: {event}: {e}")
    
    if combined_psth and combined_trials:
        # Pad all PSTH arrays to match the longest one
        combined_psth = [pad_to_match_length(psth, max_time_length) for psth in combined_psth]
        psth_combined = np.concatenate(combined_psth, axis=1)
        df_trials_combined = pd.concat(combined_trials, ignore_index=True)
        return psth_combined, df_trials_combined
    else:
        return None, None 
    




# path_to_data = '/mnt/h0/kb/Feb2025/data_behav_psth/Good_sessions/'
path_to_data = '/home/kceniabougrova/Documents/NM_project_fromIBLserver/Good_sessions'
EVENTS = ['stimOnTrigger_times']

groups = {
    "DA": ["ZFM-03447", "ZFM-04026", "ZFM-04022_R", "ZFM-03448", "ZFM-04019", "ZFM-04022_L"]}  

for group_name, group_mice in groups.items():
    for event in EVENTS:
        print(f"Processing group {group_name} | Event: {event}")
        psth_combined, df_trials_combined = load_group_data(group_mice, event, path_to_data, PERIEVENT_WINDOW=[-1, 2])
if psth_combined.shape[0] == 91:
    psth_combined = psth_combined.T  # Now shape = (10628, 91)

        


# Define the mapping for NM groups
nm_groups = {
    "DA": ["ZFM-03447", "ZFM-04026", "ZFM-04022_R", "ZFM-03448", "ZFM-04019", "ZFM-04022_L"],
}

# Create a function to map the mouse ID to NM label
def map_nm(mouse):
    for nm, mice in nm_groups.items():
        if mouse in mice:
            return nm
    return "Unknown"  # In case the mouse is not found in any group

# Add the "NM" column
df_trials_combined['NM'] = df_trials_combined['mouse'].apply(map_nm)

# Create a function to determine the session type
def determine_sessiontype(eid, probL_values):
    # Check if probL contains only 0.5, 0.2, 0.8 and no other values for a particular eid
    if set(probL_values) == {0.5, 0.2, 0.8}:
        return "BiasedCW"
    else:
        return "TrainingCW"

# Group by "eid" and apply the session type rule
df_trials_combined['sessiontype'] = df_trials_combined.groupby('eid')['probL'].transform(lambda x: determine_sessiontype(x.name, x))

# Drop the 'mouse' column and reorder the columns
df_trials_combined = df_trials_combined.drop(columns=['mouse', 'probabilityLeft'])

# Check if the number of rows in both is the same
if df_trials_combined.shape[0] == psth_combined.shape[0]:
    # Add psth_combined as a new column 'psth_values' in df_trials_combined
    df_trials_combined['psth_values'] = [list(psth_combined[i]) for i in range(len(psth_combined))]
else:
    print("Error: The number of rows in df_trials_combined and psth_combined do not match.")

# Define the desired column order
column_order = [
    'NM', 'subject', 'subject_2', 'region', 'date', 'eid', 'sessiontype', 'trialNumber',
    'intervals_0', 'stimOnTrigger_times', 'goCueTrigger_times', 'goCue_times', 'stimOn_times',
    'firstMovement_times', 'response_times', 'feedback_times', 'stimOffTrigger_times', 'stimOff_times',
    'intervals_1', 'quiescencePeriod', 'contrastLeft', 'contrastRight', 'allContrasts', 'allSContrasts',
    'choice', 'feedbackType', 'probL', 'biasShift', 'rewardVolume', 'reactionTime', 'responseTime',
    'trialTime', 'psth_values'
]

# Reorganize the columns in the specified order
df_trials_combined = df_trials_combined[column_order]

# df_trials_combined.to_csv('/home/kceniabougrova/Downloads/DA_stimOnAlignedTraces.csv', index=False)
df_trials_combined.to_parquet('/home/kceniabougrova/Downloads/DA_stimOnAlignedTraces.pqt', index=False)
