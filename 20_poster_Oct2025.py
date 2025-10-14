#%%
##############################################################################
# 20_poster_Oct2025.py
# KB code for the poster for 15102025
# ----------------------

""" 
Step by step
A. try with 1 session first 
    1. Load the data for 1 session: 
        df_trials and 1 df_nph
            only BCW sessions 
    2. Divide the trial of each session into 3: 
        i) Quiescence = StimOn - 400ms (no movement contamination) 
        ii) StimOn = 
"""

#%%
##############################################################################
##############################################################################
##############################################################################
# 1. IMPORTS
# ----------------------------------------------------------------------
import os
import numpy as np
import pandas as pd

# =========================================================
# Load BCW "good signal - eye picked" session
# =========================================================
# df_good_BCW = pd.read_excel("/home/kceniabougrova/Downloads/good_sessions_outputs/df_good_BCW.xlsx") #302 sessions
df_good_BCW = pd.read_excel("/home/kceniabougrova/Downloads/good_sessions_outputs/df_good_BCW_2ndpass.xlsx") #278 sessions - removed some sessions with smaller signal


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
# %%
# # ========================================================
# # 2. i) Quiescence = StimOn - 400ms (no movement contamination)
# # ========================================================
# # %%

# # Sampling frequency (optional, for checking)
# dt = np.median(np.diff(df_nph["times"]))
# print(f"Estimated sampling rate: {1/dt:.1f} Hz")

# # Prepare output: a list of arrays (zdFF segments per trial)
# segments = []

# for i, row in df_trials.iterrows():
#     t0 = row["stimOnTrigger_times"] - 0.4  # 400 ms before
#     t1 = row["stimOnTrigger_times"]        # event time
    
#     # Select zdFF samples between t0 and t1
#     mask = (df_nph["times"] >= t0) & (df_nph["times"] <= t1)
#     zdff_segment = df_nph.loc[mask, "zdFF"].values
    
#     segments.append(zdff_segment)

# # Optionally store in new DataFrame
# df_segments = pd.DataFrame({
#     "trial_idx": np.arange(len(df_trials)),
#     "stimOnTrigger_times": df_trials["stimOnTrigger_times"],
#     "zdFF_segment": segments
# })

# print(df_segments.head())


# df_segments["mean_preStim_zdFF"] = df_segments["zdFF_segment"].apply(np.mean)


# # ========================================================
# # Visualize
# # Option 1
# from scipy.stats import sem

# # Interpolate all segments to the same time vector
# n_samples = int(window * fs)
# aligned = np.full((len(df_segments), n_samples), np.nan)

# for i, seg in enumerate(df_segments["zdFF_segment"]):
#     if len(seg) > 0:
#         seg_interp = np.interp(np.linspace(0, 1, n_samples),
#                                np.linspace(0, 1, len(seg)), seg)
#         aligned[i, :] = seg_interp

# # Compute mean and SEM
# mean_zdff = np.nanmean(aligned, axis=0)
# sem_zdff = sem(aligned, axis=0, nan_policy="omit")

# # Plot
# plt.figure(figsize=(8, 5))
# plt.plot(-time_vector, mean_zdff, color="darkorange")
# plt.fill_between(-time_vector, mean_zdff - sem_zdff, mean_zdff + sem_zdff, alpha=0.3)
# plt.axvline(0, color="k", linestyle="--")
# plt.xlabel("Time (s)")
# plt.ylabel("zdFF (mean ± SEM)")
# plt.title("Average pre-stimulus zdFF across trials")
# plt.tight_layout()
# plt.show()

# # Option 2
# plt.figure(figsize=(6, 6))
# plt.imshow(aligned, aspect="auto", extent=[-window, 0, 0, len(aligned)],
#            cmap="inferno", origin="lower")
# plt.axvline(0, color="w", linestyle="--")
# plt.xlabel("Time (s)")
# plt.ylabel("Trial")
# plt.title("zdFF activity before stimOn (heatmap)")
# plt.colorbar(label="zdFF")
# plt.tight_layout()
# plt.show()

# %%
# ========================================================
# 1 - Define all event time intervals per trial
# ========================================================
import numpy as np
import pandas as pd

# --- Compute event-based time windows ---
df_intervals = pd.DataFrame()
df_intervals["trial_idx"] = df_trials.index

# A. Quiescence: stimOn - 0.4 → stimOn
df_intervals["quiescence_start"] = df_trials["stimOnTrigger_times"] - 0.4
df_intervals["quiescence_end"]   = df_trials["stimOnTrigger_times"]

# B. StimOn: stimOn → stimOn + 0.1
df_intervals["stim_start"] = df_trials["stimOnTrigger_times"]
df_intervals["stim_end"]   = df_trials["stimOnTrigger_times"] + 0.1

# C. Feedback: feedback → feedback + 0.5, normalized to baseline (-0.1 → 0)
df_intervals["feedback_start"] = df_trials["feedback_times"]
df_intervals["feedback_end"]   = df_trials["feedback_times"] + 0.5
df_intervals["feedback_baseline_start"] = df_trials["feedback_times"] - 0.1
df_intervals["feedback_baseline_end"]   = df_trials["feedback_times"]

# %%
# ========================================================
# 2 - Extract zdFF segments per trial per window
# ========================================================
def extract_mean(df_nph, start_times, end_times):
    """Return per-trial mean zdFF between start and end times."""
    means = []
    for t0, t1 in zip(start_times, end_times):
        mask = (df_nph["times"] >= t0) & (df_nph["times"] <= t1)
        means.append(df_nph.loc[mask, "zdFF"].mean())
    return np.array(means)

# %%
# Now apply to each interval:
# A. Quiescence mean
df_intervals["quiescence_mean"] = extract_mean(
    df_nph, df_intervals["quiescence_start"], df_intervals["quiescence_end"])

# B. StimOn slope (0–0.1s)
stim_slopes = []
for t0, t1 in zip(df_intervals["stim_start"], df_intervals["stim_end"]):
    mask = (df_nph["times"] >= t0) & (df_nph["times"] <= t1)
    segment = df_nph.loc[mask, "zdFF"].values
    if len(segment) >= 2:
        slope = (segment[-1] - segment[0]) / (t1 - t0)
    else:
        slope = np.nan
    stim_slopes.append(slope)

df_intervals["stim_slope"] = stim_slopes

# C. Feedback normalized mean
fb_baseline = extract_mean(df_nph,
                           df_intervals["feedback_baseline_start"],
                           df_intervals["feedback_baseline_end"])
fb_response = extract_mean(df_nph,
                           df_intervals["feedback_start"],
                           df_intervals["feedback_end"])
df_intervals["feedback_norm_mean"] = fb_response - fb_baseline

# %%
# ========================================================
# 3 - Combine into a per-trial summary
# ========================================================
df_summary = df_trials.copy()
df_summary["quiescence_mean_zdFF"] = df_intervals["quiescence_mean"]
df_summary["stimOn_slope"] = df_intervals["stim_slope"]
df_summary["feedback_norm_mean"] = df_intervals["feedback_norm_mean"]

# %%
# ========================================================
# 4 - Visualize the relationships
# ========================================================
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].hist(df_summary["quiescence_mean_zdFF"], bins=30, color="royalblue")
axs[0].set_title("Mean zdFF (quiescence -0.4→0s)")

axs[1].hist(df_summary["stimOn_slope"], bins=30, color="orange")
axs[1].set_title("zdFF slope (stimOn→+0.1s)")

axs[2].hist(df_summary["feedback_norm_mean"], bins=30, color="seagreen")
axs[2].set_title("Normalized mean (feedback 0→+0.5s)")

for ax in axs: ax.set_xlabel("zdFF metric"); ax.set_ylabel("Trial count")
plt.tight_layout()
plt.show()

# %%
# ========================================================
# 5 - Select/build the predictors
# ========================================================
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler


# =========================================================
    # 1. FILTER: unbiased & valid-choice trials
    # =========================================================
df_trials = df_summary.copy()
df_pred = df_trials.copy()

df_pred = df_pred[df_pred["probabilityLeft"] == 0.5]
df_pred = df_pred[df_pred["choice"] != 0].reset_index(drop=True)

# =========================================================
    # 2. DERIVED COLUMNS
    # =========================================================

# (A) current stimSide: sign of allSContrasts → -1 (ipsilateral) / +1 (contralateral)
    # =========================================================
    # Define stimSide from contrastLeft and contrastRight
    # =========================================================
def get_stim_side(row):
    if pd.notna(row["contrastLeft"]) and pd.isna(row["contrastRight"]):
        return -1  # left stimulus only
    elif pd.notna(row["contrastRight"]) and pd.isna(row["contrastLeft"]):
        return 1   # right stimulus only
    else:
        return np.nan  # ambiguous (both present or both NaN)

df_pred["stimSide"] = df_pred.apply(get_stim_side, axis=1)

# (B) previous choice
df_pred["prev_choice"] = df_pred["choice"].shift(1)

# (C) previous feedback outcome
df_pred["prev_feedbackType"] = df_pred["feedbackType"].shift(1)

# remove first trial (no previous)
df_pred = df_pred.dropna(subset=["prev_choice", "prev_feedbackType"]).reset_index(drop=True)

# =========================================================
    # 3. RESCALING (0–1 range)
    # =========================================================

# scaler = MinMaxScaler()

# cols_to_rescale = ["quiescenceTime", "quiescencePeriod", "allContrasts", "reactionTime"]
# df_pred[cols_to_rescale] = scaler.fit_transform(df_pred[cols_to_rescale])

# how to rescale allSContrasts if needed
# df_pred["allSContrasts_rescaled"] = 2 * (df_pred["allSContrasts"] - df_pred["allSContrasts"].min()) / (df_pred["allSContrasts"].max() - df_pred["allSContrasts"].min()) - 1

    # =========================================================
    # 3.B. RESCALING (0–1 for positive vars, ±1 for signed vars)
    # =========================================================

# # Positive-only features
# cols_to_rescale_01 = ["quiescenceTime", "quiescencePeriod", "reactionTime"]
# df_pred[cols_to_rescale_01] = MinMaxScaler().fit_transform(df_pred[cols_to_rescale_01])

# Signed features (like contrast)
# it was only allContrasts

# Use MaxAbsScaler in all
cols_to_rescale_pm1 = ["allContrasts", "quiescenceTime", "quiescencePeriod", "reactionTime"]
df_pred[cols_to_rescale_pm1] = MaxAbsScaler().fit_transform(df_pred[cols_to_rescale_pm1])




# =========================================================
    # 4. KEEP ONLY THE DESIRED COLUMNS
    # =========================================================
df_predictors = df_pred[[
    "quiescenceTime",
    "quiescencePeriod",
    "stimSide",
    "allContrasts",
    "reactionTime",
    "choice",
    "feedbackType",
    "prev_choice",
    "prev_feedbackType"
]]

print("✅ Predictor matrix ready!")
print(df_predictors.head())


# %%
# ========================================================
# 6 - Ridge regression 
# ========================================================

# =========================================================
    # 1. Choose the target
    # =========================================================
target = df_pred["feedback_norm_mean"]   # or "stimOn_slope", "quiescence_mean_zdFF"

# =========================================================
    # 2. Create design matrix (X) and target (y)
    # =========================================================
import numpy as np
# from sklearn.preprocessing import StandardScaler

# Define predictors (your chosen behavioral variables)
predictor_cols = [
    "quiescenceTime",
    "quiescencePeriod",
    "stimSide",
    "allContrasts",
    "reactionTime",
    "choice",
    "feedbackType",
    "prev_choice",
    "prev_feedbackType"
]

X = df_predictors[predictor_cols].values
y = target.values

# Standardize predictors for better ridge behavior
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)


# =========================================================
    # 3. Fit Ridge regression model
    # =========================================================
from sklearn.linear_model import ridge_regression
from sklearn.metrics import r2_score

# Run ridge regression with regularization alpha
alpha = 1.0
coef = ridge_regression(X_scaled, y, alpha=alpha)

# Predict values
y_pred = X_scaled @ coef
r2 = r2_score(y, y_pred)

print(f"✅ R² score: {r2:.3f}")

# =========================================================
    # 4. a) Plot results
    # =========================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
plt.hist(y, bins=30, alpha=0.5, label="Actual", color="royalblue")
plt.hist(y_pred, bins=30, alpha=0.5, label="Predicted", color="orange")
plt.xlabel("zdFF response")
plt.ylabel("Count")
plt.title(f"Actual vs Predicted — Ridge Regression (R²={r2:.2f})")
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
    # 4. b) Plot Coefficients (Weights) 
    # =========================================================
plt.figure(figsize=(5,4))
plt.bar(predictor_cols, coef, color="darkorange")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Weight (β)")
plt.title("Behavioral variable weights (Ridge Regression)")
plt.axhline(0, color="k", linestyle="--")
plt.tight_layout()
plt.show()


# =========================================================
    # 5. Cross-validation (optional)
    # =========================================================
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []

for train_idx, test_idx in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    coef = ridge_regression(X_train, y_train, alpha=alpha)
    y_pred = X_test @ coef
    r2_scores.append(r2_score(y_test, y_pred))

print(f"Mean CV R² = {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# %%














































































































#%%
# %%
# %%
##############################################################################
# RIDGE REGRESSION LOOP ACROSS SESSIONS
##############################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import ridge_regression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
df_good_BCW = pd.read_excel(os.path.join(BASE_DIR, "df_good_BCW.xlsx"))

# containers
results = []

# predictors you always use
predictor_cols = [
    "quiescenceTime","quiescencePeriod","stimSide","allSContrasts",
    "reactionTime","choice","feedbackType","prev_choice","prev_feedbackType"
]
targets = ["quiescence_mean_zdFF", "stimOn_slope", "feedback_norm_mean"]
alpha = 1.0

# -------------------------------------------------------------
def extract_mean(df_nph, start_times, end_times):
    means = []
    for t0, t1 in zip(start_times, end_times):
        mask = (df_nph["times"] >= t0) & (df_nph["times"] <= t1)
        means.append(df_nph.loc[mask, "zdFF"].mean())
    return np.array(means)

# -------------------------------------------------------------
def get_stim_side(row):
    if pd.notna(row["contrastLeft"]) and pd.isna(row["contrastRight"]):
        return -1
    elif pd.notna(row["contrastRight"]) and pd.isna(row["contrastLeft"]):
        return 1
    else:
        return np.nan

# -------------------------------------------------------------
FLIP_FIBERS = {"probe09", "probe12", "probe16", "probe19", "probe24", "probe28", "probe31", "probe33"}

# get current session fiber
fiber = row["fiber"]
flip = -1 if fiber in FLIP_FIBERS else 1

def base_stim_side(row):
    # Left-only present (right is NaN)  -> -1
    if pd.notna(row["contrastLeft"]) and pd.isna(row["contrastRight"]):
        return -1
    # Right-only present (left is NaN)  -> +1
    elif pd.notna(row["contrastRight"]) and pd.isna(row["contrastLeft"]):
        return 1
    # both present or both NaN -> ambiguous
    else:
        return np.nan

# -------------------------------------------------------------
for i, row in df_good_BCW.iterrows():
    subject, date, region, eid, fiber = row["subject"], str(row["date"])[:10], row["region"], row["eid"], row['fiber']
    print(f"\n📦 Session {i}: {subject} | {date} | {region} | {fiber}")

    # -------- find matching files
    df_trials_file = [f for f in os.listdir(BASE_DIR)
                      if f.startswith("df_trials_") and subject in f and date in f and region in f and eid in f]
    df_nph_file = [f for f in os.listdir(BASE_DIR)
                   if f.startswith("df_nph_") and subject in f and date in f and region in f and eid in f]
    if not df_trials_file or not df_nph_file:
        print("⚠️ missing files, skipping.")
        continue

    df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
    df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

    # -------- event windows
    df_int = pd.DataFrame({"trial_idx": df_trials.index})
    df_int["q_start"], df_int["q_end"] = df_trials["stimOnTrigger_times"]-0.4, df_trials["stimOnTrigger_times"]
    df_int["s_start"], df_int["s_end"] = df_trials["stimOnTrigger_times"], df_trials["stimOnTrigger_times"]+0.15 #0.1 has nans (only 1 value within that interval or so)
    df_int["f_start"], df_int["f_end"] = df_trials["feedback_times"], df_trials["feedback_times"]+0.5
    df_int["f_bstart"], df_int["f_bend"] = df_trials["feedback_times"]-0.1, df_trials["feedback_times"]

    # -------- compute zdFF metrics
    df_summary = df_trials.copy()
    df_summary["quiescence_mean_zdFF"] = extract_mean(df_nph, df_int["q_start"], df_int["q_end"])
    stim_slopes = []
    for t0, t1 in zip(df_int["s_start"], df_int["s_end"]):
        seg = df_nph.loc[(df_nph["times"]>=t0)&(df_nph["times"]<=t1),"zdFF"].values
        slope = (seg[-1]-seg[0])/(t1-t0) if len(seg)>=2 else np.nan
        stim_slopes.append(slope)
    df_summary["stimOn_slope"] = stim_slopes
    fb_base = extract_mean(df_nph, df_int["f_bstart"], df_int["f_bend"])
    fb_resp = extract_mean(df_nph, df_int["f_start"], df_int["f_end"])
    df_summary["feedback_norm_mean"] = fb_resp - fb_base

    # -------- filter + derived predictors
    # df_pred = df_summary[(df_summary["probabilityLeft"]==0.5) & (df_summary["choice"]!=0)].copy()
    df_pred = df_summary[
        (df_summary["probabilityLeft"] == 0.5)
        & (df_summary["choice"] != 0)
        & (df_summary["firstMovement_times"].notna())
    ].copy()
    # compute stimSide then flip if fiber is in the list
    df_pred["stimSide"] = df_pred.apply(base_stim_side, axis=1)
    df_pred["stimSide"] = df_pred["stimSide"] * flip
    df_pred["prev_choice"] = df_pred["choice"].shift(1)
    df_pred["prev_feedbackType"] = df_pred["feedbackType"].shift(1)
    df_pred.dropna(subset=["prev_choice","prev_feedbackType","stimSide"], inplace=True)

    # -------- scale predictors
    scaler = MaxAbsScaler()
    df_pred[predictor_cols] = scaler.fit_transform(df_pred[predictor_cols])

    X = df_pred[predictor_cols].values

    # -------- loop over the 3 neural targets
    for target_name in targets:
        y = df_pred[target_name].values
        if np.all(np.isnan(y)):
            continue

        # Fit ridge
        coef = ridge_regression(X, y, alpha=alpha)
        y_pred = X @ coef
        r2 = r2_score(y, y_pred)

        # CV R²
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_cv = []
        for tr, ts in kf.split(X):
            coef_cv = ridge_regression(X[tr], y[tr], alpha=alpha)
            y_pred_cv = X[ts] @ coef_cv
            r2_cv.append(r2_score(y[ts], y_pred_cv))
        mean_cv, std_cv = np.mean(r2_cv), np.std(r2_cv)

        print(f"   {target_name}: R²={r2:.3f}, CV={mean_cv:.3f}±{std_cv:.3f}")

        # store
        results.append({
            "session_idx": i,
            "subject": subject,
            "date": date,
            "region": region,
            "fiber": fiber, 
            "target": target_name,
            "r2": r2,
            "r2_cv_mean": mean_cv,
            "r2_cv_std": std_cv,
            **{f"β_{col}": b for col,b in zip(predictor_cols, coef)}
        })

        # plot weights for each session
        plt.figure(figsize=(5,4))
        plt.bar(predictor_cols, coef, color="darkorange")
        plt.xticks(rotation=45, ha="right")
        plt.axhline(0, color="k", linestyle="--")
        plt.ylabel("Weight (β)")
        plt.title(f"{target_name} — {subject} {date} {fiber} ({region})\nR²={r2:.2f}, CV={mean_cv:.2f}")
        plt.tight_layout()
        plt.show()

# %%
# SAVE RESULTS
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(BASE_DIR, f"ridge_results_allSessions_alpha{alpha}_signed.csv"), index=False)

# Plot distribution of R² per signal
plt.figure(figsize=(7,4))
for t in targets:
    plt.hist(df_results[df_results["target"]==t]["r2"], bins=20, alpha=0.5, label=t)
plt.xlabel("R²")
plt.ylabel("Sessions")
plt.title(f"Explained variance of neural signals by behavioral model (α={alpha})")
plt.legend()
plt.tight_layout()
plt.show()

# %%





# %%
# ===============================================================================
# ===============================================================================
# ===============================================================================
# to choose alpha
# ===============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ridge_regression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import r2_score

BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
df_good_BCW = pd.read_excel(os.path.join(BASE_DIR, "df_good_BCW.xlsx"))

alphas = [0.1, 1, 5, 10, 50]

targets = ["quiescence_mean_zdFF", "stimOn_slope", "feedback_norm_mean"]

predictor_cols = [
    "quiescenceTime", "quiescencePeriod", "stimSide", "allContrasts",
    "reactionTime", "choice", "feedbackType", "prev_choice", "prev_feedbackType"
]
flip_fibers = {"probe09","probe12","probe16","probe19","probe24","probe28","probe31","probe33"}

# =========================================================
# Helper functions
# =========================================================
def extract_mean(df, s, e):
    out = np.full(len(s), np.nan)
    for i, (t0, t1) in enumerate(zip(s, e)):
        mask = (df["times"] >= t0) & (df["times"] <= t1)
        if mask.any():
            out[i] = df.loc[mask, "zdFF"].mean()
    return out

def base_stim_side(r):
    if pd.notna(r["contrastLeft"]) and pd.isna(r["contrastRight"]):
        return -1
    elif pd.notna(r["contrastRight"]) and pd.isna(r["contrastLeft"]):
        return 1
    else:
        return np.nan

# =========================================================
# STEP 1 — Precompute all metrics once
# =========================================================
cached_sessions = []
df_subset = df_good_BCW.sample(n=50, random_state=42)

for i, row in df_subset.iterrows():
    print(i,"==================================================================")
    subject, date, region, eid, fiber = row["subject"], str(row["date"])[:10], row["region"], row["eid"], row["fiber"]

    df_trials_file = [f for f in os.listdir(BASE_DIR)
                      if f.startswith("df_trials_") and subject in f and date in f and region in f and eid in f]
    df_nph_file = [f for f in os.listdir(BASE_DIR)
                   if f.startswith("df_nph_") and subject in f and date in f and region in f and eid in f]
    if not df_trials_file or not df_nph_file:
        continue

    df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
    df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

    # time intervals
    stim = df_trials["stimOnTrigger_times"]
    fb = df_trials["feedback_times"]

    df_int = pd.DataFrame({
        "q_start": stim - 0.4, "q_end": stim,
        "s_start": stim, "s_end": stim + 0.1,
        "f_start": fb, "f_end": fb + 0.5,
        "f_bstart": fb - 0.1, "f_bend": fb
    })
    print("step2")
    # compute zdFF metrics
    q = extract_mean(df_nph, df_int["q_start"], df_int["q_end"])
    s = np.full(len(stim), np.nan)
    for j, (t0, t1) in enumerate(zip(df_int["s_start"], df_int["s_end"])):
        seg = df_nph.loc[(df_nph["times"]>=t0)&(df_nph["times"]<=t1),"zdFF"].values
        if len(seg) >= 2:
            s[j] = (seg[-1]-seg[0])/(t1-t0)
    fb_base = extract_mean(df_nph, df_int["f_bstart"], df_int["f_bend"])
    fb_resp = extract_mean(df_nph, df_int["f_start"], df_int["f_end"])
    f = fb_resp - fb_base

    # create summary
    df_summary = df_trials.copy()
    df_summary["quiescence_mean_zdFF"] = q
    df_summary["stimOn_slope"] = s
    df_summary["feedback_norm_mean"] = f

    cached_sessions.append((df_summary, fiber, subject, date, region))
    print("end loop",i)
print(f"✅ Cached {len(cached_sessions)} sessions")

# =========================================================
# STEP 2 — Loop over α values
# =========================================================
results_alpha = []

for alpha in alphas:
    r2_all = []
    print(f"\n===== α = {alpha:.3f} =====")

    for df_summary, fiber, subject, date, region in cached_sessions:
        # filters
        df_pred = df_summary[
            (df_summary["probabilityLeft"]==0.5)
            & (df_summary["choice"]!=0)
            & (df_summary["firstMovement_times"].notna())
        ].copy()

        # stimSide + history
        df_pred["stimSide"] = df_pred.apply(base_stim_side, axis=1)
        df_pred["stimSide"] *= (-1 if fiber in flip_fibers else 1)
        df_pred["prev_choice"] = df_pred["choice"].shift(1)
        df_pred["prev_feedbackType"] = df_pred["feedbackType"].shift(1)
        df_pred.dropna(subset=["stimSide","prev_choice","prev_feedbackType"], inplace=True)
        df_pred.dropna(subset=predictor_cols, inplace=True)

        # scaling
        if len(df_pred) < 10:
            continue
        scaler = MaxAbsScaler()
        df_pred[predictor_cols] = scaler.fit_transform(df_pred[predictor_cols])
        X = df_pred[predictor_cols].to_numpy()

        # loop through targets
        print("entering another loop")
        for t in targets:
            y = df_pred[t].to_numpy()
            mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
            Xc, yc = X[mask], y[mask]
            if len(yc) < 10:
                continue
            coef = ridge_regression(Xc, yc, alpha=alpha)
            y_pred = Xc @ coef
            r2_all.append(r2_score(yc, y_pred))

    mean_r2, std_r2 = np.mean(r2_all), np.std(r2_all)
    print(f"✅ α={alpha:.3f}: mean R²={mean_r2:.3f} ± {std_r2:.3f}")
    results_alpha.append({"alpha": alpha, "mean_r2": mean_r2, "std_r2": std_r2})

# =========================================================
# STEP 3 — Plot R² vs α
# =========================================================
df_alpha = pd.DataFrame(results_alpha)
plt.figure(figsize=(6,4))
plt.semilogx(df_alpha["alpha"], df_alpha["mean_r2"], "o-", color="royalblue")
plt.fill_between(df_alpha["alpha"],
                 df_alpha["mean_r2"] - df_alpha["std_r2"],
                 df_alpha["mean_r2"] + df_alpha["std_r2"],
                 color="royalblue", alpha=0.2)
plt.xlabel("α (regularization strength)")
plt.ylabel("Mean R² (across all sessions)")
plt.title("Ridge α tuning (fast cached version)")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()





























#%%
# ==============================================================================
# ==============================================================================
# ==============================================================================
# SPLIT BY NMS the regression results 
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# LOAD AND MAP NEUROMODULATORS
# =========================================================
# df = pd.read_csv("/home/kceniabougrova/Downloads/good_sessions_outputs/ridge_results_allSessions_alpha1.0.csv")
# df = pd.read_csv("/home/kceniabougrova/Downloads/good_sessions_outputs/ridge_results_allSessions_alpha1.0_unsigned_noTimeVars.csv")
df = pd.read_csv("/home/kceniabougrova/Downloads/good_sessions_outputs/regression_results_allSessions_alpha1.0_unsigned_noTimeVars.csv") 

nm_map = {
    # DA
    "ZFM-03447": "DA", "ZFM-03448": "DA", "ZFM-03450": "DA",
    "ZFM-04019": "DA", "ZFM-04022": "DA", "ZFM-04026": "DA",
    # 5-HT
    "ZFM-03061": "5-HT", "ZFM-03065": "5-HT", "ZFM-04392": "5-HT", 
    'ZFM-03059': "5-HT", 'ZFM-03062': "5-HT",
    'ZFM-05236': "5-HT", 'ZFM-05248': "5-HT", 'ZFM-05245': "5-HT", 'ZFM-05235': "5-HT",
    # NE
    "ZFM-06268": "NE", "ZFM-06271": "NE", "ZFM-06272": "NE",
    "ZFM-06171": "NE", "ZFM-06275": "NE",
    'ZFM-04533': "NE", 'ZFM-04534': "NE",
    # ACh
    "ZFM-06305": "ACh", "ZFM-06946": "ACh", "ZFM-06948": "ACh"
}

df["NM"] = df["subject"].map(nm_map)
df = df.dropna(subset=["NM"])  # keep only mapped mice

print(df["NM"].value_counts())

# =========================================================
# 1️⃣ SUMMARY BY NM AND TARGET
# =========================================================
summary = (
    df.groupby(["NM", "target"])
      .agg(mean_r2=("r2", "mean"),
           std_r2=("r2", "std"),
           mean_cv=("r2_cv_mean", "mean"),
           std_cv=("r2_cv_mean", "std"))
      .reset_index()
)

print(summary)

# =========================================================
# PLOT 1 — Mean R² per NM × target
# =========================================================
plt.figure(figsize=(8,5))
sns.barplot(data=summary, x="NM", y="mean_r2", hue="target",
            palette="Set2", capsize=.1, errwidth=1)
plt.ylabel("Mean R² (ridge α=1)")
plt.title("Explained variance per Neuromodulator and target")
plt.legend(title="Target", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# =========================================================
# 2️⃣ PLOT 2 — Cross-validated R² per NM × target
# =========================================================
plt.figure(figsize=(8,5))
sns.barplot(data=summary, x="NM", y="mean_cv", hue="target",
            palette="coolwarm", capsize=.1)
plt.ylabel("Mean CV R²")
plt.title("Cross-validated R² per Neuromodulator and target")
plt.axhline(0, color='k', linestyle='--', lw=1)
plt.legend(title="Target", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# =========================================================
# Define your custom palette
nm_colors = {
    "DA": "#BC1717",        # strong red
    "5-HT": "#7855F9",      # dark slate/purple
    "NE": "#4984F3",        # cornflower blue
    "ACh": "#167D16"        # forest green (deep, not neon)
}

# =========================================================
# 3️⃣ PLOT 3A — Predictor weights per NM
# =========================================================
# melt weight columns (β_xxx)
coef_cols = [c for c in df.columns if c.startswith("β_")]
df_melt = df.melt(id_vars=["NM", "target"], value_vars=coef_cols,
                  var_name="predictor", value_name="weight")
df_melt["predictor"] = df_melt["predictor"].str.replace("β_", "")

plt.figure(figsize=(10,5))
sns.boxplot(data=df_melt, x="predictor", y="weight", hue="NM",
            palette="Set3", showfliers=False)
plt.xticks(rotation=45, ha="right")
plt.axhline(0, color="k", linestyle="--", lw=1)
plt.title("Predictor weights per Neuromodulator")
plt.ylabel("Ridge β (α=1)")
plt.tight_layout()
plt.show()

# =========================================================
# 4️⃣ PLOT 3B — Predictor weights per NM and target
# =========================================================
for target_name in ["quiescence_mean_zdFF", "stimOn_slope", "feedback_norm_mean"]:
    subset = df[df["target"] == target_name]
    plt.figure(figsize=(10,5))
    sns.boxplot(
        data=subset.melt(id_vars=["NM"], value_vars=[c for c in subset.columns if c.startswith("β_")]),
        x="variable", y="value", hue="NM", palette=nm_colors
    )
    plt.title(f"Predictor weights for {target_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# %% 









































#%%
# # ========================================================
# # ========================================================
# # ========================================================
# # to see which sessions have a smaller psth
# # ========================================================
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # =========================================================
# # SETTINGS
# # =========================================================
# BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
# df_good_BCW = pd.read_excel(os.path.join(BASE_DIR, "df_good_BCW.xlsx"))

# window = [-1, 2]   # peri-feedback window (s)
# bin_size = 0.05
# time_axis = np.arange(window[0], window[1], bin_size)

# # store weak sessions
# weak_sessions = []

# # =========================================================
# # HELPER FUNCTION
# # =========================================================
# def extract_aligned_signal(df_nph, event_times, win, bin_size):
#     """Return mean and SEM of zdFF aligned to event_times."""
#     n_bins = int((win[1] - win[0]) / bin_size)
#     aligned = np.full((len(event_times), n_bins), np.nan)

#     for i, t_event in enumerate(event_times):
#         t0, t1 = t_event + win[0], t_event + win[1]
#         seg = df_nph.loc[(df_nph["times"] >= t0) & (df_nph["times"] < t1), ["times", "zdFF"]].copy()
#         if len(seg) > 5:
#             seg["t_rel"] = seg["times"] - t_event
#             aligned[i, :] = np.interp(time_axis, seg["t_rel"], seg["zdFF"], left=np.nan, right=np.nan)
#     mean_sig = np.nanmean(aligned, axis=0)
#     sem_sig = np.nanstd(aligned, axis=0) / np.sqrt(np.sum(~np.isnan(aligned[:, 0])))
#     return mean_sig, sem_sig


# # =========================================================
# # LOOP THROUGH SESSIONS
# # =========================================================
# for i, row in df_good_BCW[199:].iterrows():
#     subject, date, region, eid, fiber = row["subject"], str(row["date"])[:10], row["region"], row["eid"], row["fiber"]
#     print(f"\n📦 Session {i}: {subject} | {date} | {region} | {fiber}")

#     # --- find matching files
#     df_trials_file = [f for f in os.listdir(BASE_DIR)
#                       if f.startswith("df_trials_") and subject in f and date in f and region in f and eid in f]
#     df_nph_file = [f for f in os.listdir(BASE_DIR)
#                    if f.startswith("df_nph_") and subject in f and date in f and region in f and eid in f]
#     if not df_trials_file or not df_nph_file:
#         print("⚠️ Missing files — skipping")
#         continue

#     df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
#     df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

#     # --- separate correct/incorrect feedback
#     corr = df_trials[df_trials["feedbackType"] == 1]["feedback_times"].dropna().values
#     incorr = df_trials[df_trials["feedbackType"] == -1]["feedback_times"].dropna().values
#     if len(corr) < 5 or len(incorr) < 5:
#         print("⚠️ Not enough trials — skipping")
#         continue

#     # --- extract signals
#     mean_corr, sem_corr = extract_aligned_signal(df_nph, corr, window, bin_size)
#     mean_incorr, sem_incorr = extract_aligned_signal(df_nph, incorr, window, bin_size)

#     # =========================================================
#     # FILTER BY MAX RESPONSE
#     # =========================================================
#     post_mask = (time_axis >= 0) & (time_axis <= 2)
#     max_corr = np.nanmax(mean_corr[post_mask]) if np.any(~np.isnan(mean_corr[post_mask])) else np.nan
#     max_incorr = np.nanmax(mean_incorr[post_mask]) if np.any(~np.isnan(mean_incorr[post_mask])) else np.nan
#     max_response = np.nanmax([max_corr, max_incorr])

#     # ✅ only plot and log if response is weak (<0.4)
#     if np.isnan(max_response) or max_response >= 0.4:
#         print(f"🚫 Max response = {max_response:.3f} ≥ 0.4 — skipping plot")
#         continue

#     print(f"✅ Weak response (max={max_response:.2f}) — plotting and logging")
#     weak_sessions.append({
#         "index": i,
#         "subject": subject,
#         "date": date,
#         "region": region,
#         "fiber": fiber,
#         "max_response": max_response
#     })

#     # =========================================================
#     # PLOT
#     # =========================================================
#     plt.figure(figsize=(6, 4))
#     plt.plot(time_axis, mean_corr, color="seagreen", lw=2, label="Correct")
#     plt.fill_between(time_axis, mean_corr - sem_corr, mean_corr + sem_corr, color="seagreen", alpha=0.3)
#     plt.plot(time_axis, mean_incorr, color="firebrick", lw=2, label="Incorrect")
#     plt.fill_between(time_axis, mean_incorr - sem_incorr, mean_incorr + sem_incorr, color="firebrick", alpha=0.3)
#     plt.axvline(0, color="k", linestyle="--", lw=1)
#     plt.xlabel("Time from feedback (s)")
#     plt.ylabel("zdFF")
#     plt.title(f"{subject} | {region} | {fiber} | {date}\nMax zdFF = {max_response:.2f}")
#     plt.ylim([-1, 2])
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# # =========================================================
# # SUMMARY OUTPUT
# # =========================================================
# # if weak_sessions:
# #     df_weak = pd.DataFrame(weak_sessions)
# #     print("\n🧩 Weak sessions found (<0.4 max zdFF):")
# #     print(df_weak[["index", "subject", "region", "fiber", "max_response"]])
# #     # Optionally save:
# #     # df_weak.to_csv(os.path.join(BASE_DIR, "weak_sessions_below0.4.csv"), index=False)
# # else:
# #     print("\n✅ No weak sessions (all > 0.4 ΔF/F)")




#%% 







































#%%
# ===============================================================
# ===============================================================
# ===============================================================
# plot PSTHs for each sessiom
# ===============================================================
# =========================================================
# LOOP THROUGH ALL SESSIONS AND PLOT BY NM
# =========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
df_good_BCW = pd.read_excel(os.path.join(BASE_DIR, "df_good_BCW_2ndpass.xlsx"))
PERIEVENT_WINDOW = [-1, 2]
baseline_window = [-0.1, 0]
EVENT = "feedback_times"  # or "stimOnTrigger_times"
SAVE = False

# =========================================================
# COLORS PER NM (exact style from your reference)
# =========================================================
nm_colors = {
    "DA": ["#f2d8d5", "#f7b7ae", "#ec8072", "#d44836", "#a1271a"],  # lighter → darker
    "5-HT": ["#ddd3f2", "#c5a5f0", "#a374e8", "#7f4cc8", "#572f91"],
    "NE": ["#d3e8f5", "#a3d0f2", "#6eb7ec", "#368cd4", "#166ea3"],
    "ACh": ["#d7f1d4", "#aceca8", "#76d873", "#46b944", "#2f7a2d"]
}

contrast_labels = {0.0: "0", 0.0625: "6", 0.125: "12", 0.25: "25", 0.5: "50", 1.0: "100"}

# =========================================================
# FUNCTION TO ALIGN AND BASELINE-CORRECT
# =========================================================
def extract_aligned(df_nph, event_times, perievent_window=PERIEVENT_WINDOW, baseline_window=baseline_window):
    # Compute frame rate dynamically from df_nph
    fr = 1.0 / np.median(np.diff(df_nph["times"]))
    time_axis = np.arange(perievent_window[0], perievent_window[1], 1/fr)

    aligned = []
    for t in event_times:
        mask = (df_nph["times"] >= t + perievent_window[0]) & (df_nph["times"] <= t + perievent_window[1])
        segment = df_nph.loc[mask]
        if len(segment) < 10:
            continue

        # relative time and interpolation
        segment["t_rel"] = segment["times"] - t
        interp = np.interp(time_axis, segment["t_rel"], segment["zdFF"])

        # baseline correction (e.g., -0.5 to 0)
        base_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
        baseline = np.nanmean(interp[base_mask])
        aligned.append(interp - baseline)

    return np.array(aligned), time_axis


# # =========================================================
# # PREPARE TIME AXIS
# # =========================================================
# # time_axis = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1],
# #                         int((PERIEVENT_WINDOW[1] - PERIEVENT_WINDOW[0]) * 1000))
# fr = 1.0 / np.median(np.diff(df_nph["times"]))
# time_axis = np.arange(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], 1/fr)

# =========================================================
# LOOP THROUGH NMs
# =========================================================
for nm in ["DA", "5-HT", "NE", "ACh"]:
    print(f"\n🎨 Plotting {nm} sessions...")
    nm_rows = df_good_BCW[df_good_BCW["NM"] == nm]

    for idx, row in nm_rows.iterrows():
        subject, date, region, fiber, eid = row["subject"], str(row["date"])[:10], row["region"], row["fiber"], row["eid"]
        print(f"   📦 {idx} — {subject} | {date} | {region} | {fiber}")

        # find files
        df_trials_file = [f for f in os.listdir(BASE_DIR)
                          if f.startswith("df_trials_") and subject in f and date in f and region in f and eid in f]
        df_nph_file = [f for f in os.listdir(BASE_DIR)
                       if f.startswith("df_nph_") and subject in f and date in f and region in f and eid in f]
        if not df_trials_file or not df_nph_file:
            print("⚠️ missing data, skipping")
            continue

        df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
        df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

        contrasts = np.sort(df_trials["allContrasts"].dropna().unique())[::-1]
        colors = nm_colors[nm][-len(contrasts):]  # match number of contrasts

        # ---------------------------------------------
        # PLOT PER SESSION (correct vs incorrect)
        # ---------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for col, (label, val) in enumerate(zip(["Correct", "Incorrect"], [1, -1])):
            ax = axes[col]
            total_trials = 0

            for contrast, color in zip(contrasts, colors):
                idx = (df_trials["allContrasts"] == contrast) & (df_trials["feedbackType"] == val)
                event_times = df_trials.loc[idx, EVENT].dropna().values
                if len(event_times) < 3:
                    continue

                aligned, time_axis = extract_aligned(df_nph, event_times)
                if aligned.size == 0:
                    continue

                mean_trace = np.nanmean(aligned, axis=0)
                sem_trace = sem(aligned, axis=0, nan_policy="omit")
                total_trials += len(event_times)

                ax.plot(time_axis, mean_trace, color=color, lw=2)
                ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.15)

            ax.axvline(0, color="black", linestyle="--", lw=1.8)
            ax.set_xlim(PERIEVENT_WINDOW)
            ax.set_ylim([-1, 2])
            ax.set_title(label, fontsize=14)
            if col == 0:
                ax.set_ylabel("ΔF/F (baseline-corrected)", fontsize=12)
            ax.set_xlabel("Time since feedback (s)", fontsize=12)
            ax.text(0.02, 0.95, f"{total_trials} trials", transform=ax.transAxes,
                    fontsize=10, va="top", color="black")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Legend (once per NM)
        patches = [plt.Line2D([0], [0], color=c, lw=2, label=f"Contrast {contrast_labels.get(k, k)}")
                   for k, c in zip(contrasts, colors)]
        fig.legend(handles=patches[::-1], frameon=False, fontsize=10,
                   loc="upper right", bbox_to_anchor=(1, 0.95))

        plt.suptitle(f"{nm} | {subject} | {date} | {region} | {fiber}", fontsize=14, y=1.03)
        plt.tight_layout()

        if SAVE:
            outdir = os.path.join(BASE_DIR, "plots_byNM")
            os.makedirs(outdir, exist_ok=True)
            fname = f"{outdir}/{nm}_{subject}_{date}_{fiber}_{EVENT}.png"
            plt.savefig(fname, dpi=300)
            print(f"💾 saved {fname}")
        else:
            plt.show()














#%% 
































































#%% 
# ==============================================================================
# ==============================================================================
# ==============================================================================
# =========================================================
# SAVE PSTH ARRAYS PER SESSION (for reuse) #DO IT FOR ONE EVENT AND THEN THE OTHER ONE
# "2..." is after changing the way FR is calculated 
# =========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
SAVE_DIR = os.path.join(BASE_DIR, "psth_arrays")
os.makedirs(SAVE_DIR, exist_ok=True)

df_good_BCW = pd.read_excel(os.path.join(BASE_DIR, "df_good_BCW_2ndpass.xlsx"))
PERIEVENT_WINDOW = [-1, 2]
baseline_window = [-0.1, 0]
EVENT = "feedback_times"  # can change to "stimOnTrigger_times"

# =========================================================
# FUNCTION TO ALIGN AND BASELINE-CORRECT
# =========================================================
def extract_aligned(df_nph, event_times, perievent_window=PERIEVENT_WINDOW, baseline_window=baseline_window):
    """Aligns zdFF to event times using the real sampling rate from df_nph."""
    fr = 1.0 / np.median(np.diff(df_nph["times"]))  # true frame rate
    time_axis = np.arange(perievent_window[0], perievent_window[1], 1/fr)

    aligned = []
    for t in event_times:
        mask = (df_nph["times"] >= t + perievent_window[0]) & (df_nph["times"] <= t + perievent_window[1])
        seg = df_nph.loc[mask]
        if len(seg) < 10:
            continue
        seg = seg.copy()
        seg["t_rel"] = seg["times"] - t
        interp = np.interp(time_axis, seg["t_rel"], seg["zdFF"])
        base_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
        baseline = np.nanmean(interp[base_mask])
        aligned.append(interp - baseline)

    return np.array(aligned), time_axis, fr

# =========================================================
# LOOP THROUGH ALL SESSIONS
# =========================================================
for i, row in df_good_BCW.iterrows():
    subject, date, region, fiber, eid, nm = row["subject"], str(row["date"])[:10], row["region"], row["fiber"], row["eid"], row["NM"]
    print(f"\n📦 Session {i}: {subject} | {date} | {region} | {fiber} | {nm}")

    # locate files
    df_trials_file = [f for f in os.listdir(BASE_DIR)
                      if f.startswith("df_trials_") and subject in f and date in f and region in f and eid in f]
    df_nph_file = [f for f in os.listdir(BASE_DIR)
                   if f.startswith("df_nph_") and subject in f and date in f and region in f and eid in f]

    if not df_trials_file or not df_nph_file:
        print("⚠️ Missing files, skipping")
        continue

    df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
    df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

    contrasts = np.sort(df_trials["allContrasts"].dropna().unique())[::-1]
    psth_mean = {}
    psth_sem = {}
    trial_counts = {}

    # ---------------------------------------------
    # CALCULATE PSTH PER CONTRAST (correct + incorrect)
    # ---------------------------------------------
    for contrast in contrasts:
        psth_mean[contrast] = {}
        psth_sem[contrast] = {}
        trial_counts[contrast] = {}

        for label, fb_type in zip(["correct", "incorrect"], [1, -1]):
            idx = (df_trials["allContrasts"] == contrast) & (df_trials["feedbackType"] == fb_type)
            event_times = df_trials.loc[idx, EVENT].dropna().values
            if len(event_times) < 3:
                psth_mean[contrast][label] = np.full_like(time_axis, np.nan)
                psth_sem[contrast][label] = np.full_like(time_axis, np.nan)
                trial_counts[contrast][label] = 0
                continue

            aligned, time_axis, fr = extract_aligned(df_nph, event_times)
            if aligned.size == 0:
                psth_mean[contrast][label] = np.full_like(time_axis, np.nan)
                psth_sem[contrast][label] = np.full_like(time_axis, np.nan)
                trial_counts[contrast][label] = 0
                continue

            psth_mean[contrast][label] = np.nanmean(aligned, axis=0)
            psth_sem[contrast][label] = sem(aligned, axis=0, nan_policy="omit")
            trial_counts[contrast][label] = len(event_times)

    # ---------------------------------------------
    # SAVE ARRAYS TO FILE
    # ---------------------------------------------
    save_name = f"2_{subject}_{date}_{region}_{fiber}_{EVENT}.npz"
    save_path = os.path.join(SAVE_DIR, save_name)

    np.savez_compressed(
        save_path,
        time_axis=time_axis,
        psth_mean=psth_mean,
        psth_sem=psth_sem,
        trial_counts=trial_counts,
        subject=subject,
        date=date,
        region=region,
        fiber=fiber,
        NM=nm,
        event=EVENT,
        frame_rate=fr
    )

    print(f"💾 Saved PSTH arrays → {save_path}")
# %%








































#%%
# ===============================================================
# ===============================================================
# ===============================================================
# PLOT PSTHs FOR A SINGLE SESSION (to check the saved arrays)
# ===============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
i = 58  # select session index here

row = df_good_BCW.iloc[i]
subject, date, region, fiber, eid = row["subject"], str(row["date"])[:10], row["region"], row["fiber"], row["eid"]
print(f"📦 Session {i} — {subject} | {date} | {region} | {fiber} | {eid}")

# Locate files
df_trials_file = [f for f in os.listdir(BASE_DIR)
                  if f.startswith("df_trials_") and subject in f and date in f and region in f and eid in f]
df_nph_file = [f for f in os.listdir(BASE_DIR)
               if f.startswith("df_nph_") and subject in f and date in f and region in f and eid in f]

if not df_trials_file or not df_nph_file:
    raise FileNotFoundError("Missing one or both data files!")

df_trials = pd.read_csv(os.path.join(BASE_DIR, df_trials_file[0]))
df_nph = pd.read_csv(os.path.join(BASE_DIR, df_nph_file[0]))

# =========================================================
# PARAMETERS
# =========================================================
win = [-1, 2]         # peri-event window (s)
fr = 1.0 / np.median(np.diff(df_nph["times"]))
bin_size = 1 / fr
time_axis = np.arange(win[0], win[1], bin_size)
print(f"📊 Frame rate: {fr:.2f} Hz  |  {len(time_axis)} samples per trial")
baseline_window = [-0.1, 0]  # baseline for subtraction
contrasts = sorted(df_trials["allContrasts"].dropna().unique())

# =========================================================
# FUNCTION TO ALIGN TRACES
# =========================================================
def extract_aligned(df_nph, event_times, win, bin_size):
    time_axis = np.arange(win[0], win[1], bin_size)
    n_bins = len(time_axis)
    aligned = np.full((len(event_times), n_bins), np.nan)

    for j, t0 in enumerate(event_times):
        t_start, t_end = t0 + win[0], t0 + win[1]
        seg = df_nph.loc[(df_nph["times"] >= t_start) & (df_nph["times"] <= t_end), ["times", "zdFF"]]
        if len(seg) > 10:
            seg = seg.copy()
            seg["t_rel"] = seg["times"] - t0
            interp_values = np.interp(time_axis, seg["t_rel"], seg["zdFF"], left=np.nan, right=np.nan)
            aligned[j, :] = interp_values
    return aligned


# =========================================================
# FUNCTION TO COMPUTE BASELINE-CORRECTED MEAN ± SEM
# =========================================================
def baseline_correct(traces, time_axis, baseline_window):
    base_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
    baseline = np.nanmean(traces[:, base_mask], axis=1, keepdims=True)
    corrected = traces - baseline
    return np.nanmean(corrected, axis=0), sem(corrected, axis=0, nan_policy="omit")

# =========================================================
# PLOT — stimOnTrigger_times (future feedback)
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
colors = plt.cm.Reds(np.linspace(0.3, 1, len(contrasts)))

for col_idx, (fb_label, fb_value) in enumerate(zip(["Future Correct", "Future Incorrect"], [1, -1])):
    ax = axes[col_idx]
    for c, color in zip(contrasts, colors):
        idx = (df_trials["allContrasts"] == c)
        next_fb = df_trials["feedbackType"].shift(-1)  # future feedback
        idx &= (next_fb == fb_value)
        events = df_trials.loc[idx, "stimOnTrigger_times"].dropna().values
        if len(events) < 5:
            continue
        aligned = extract_aligned(df_nph, events, win, bin_size)
        mean_trace, sem_trace = baseline_correct(aligned, time_axis, baseline_window)
        ax.plot(time_axis, mean_trace, color=color, lw=2, label=f"Contrast {c}")
        ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.3)
    ax.axvline(0, color="k", linestyle="--")
    ax.set_title(fb_label)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(win)
    if col_idx == 0:
        ax.set_ylabel("ΔF/F (baseline-corrected)")
    ax.legend(fontsize=8)

fig.suptitle(f"{subject} | {region} | {fiber} | stimOnTrigger_times (by future feedback)", fontsize=12)
plt.tight_layout()
plt.show()

# =========================================================
# PLOT — feedback_times (current feedback)
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for col_idx, (fb_label, fb_value) in enumerate(zip(["Correct", "Incorrect"], [1, -1])):
    ax = axes[col_idx]
    for c, color in zip(contrasts, colors):
        idx = (df_trials["allContrasts"] == c) & (df_trials["feedbackType"] == fb_value)
        events = df_trials.loc[idx, "feedback_times"].dropna().values
        if len(events) < 5:
            continue
        aligned = extract_aligned(df_nph, events, win, bin_size)
        mean_trace, sem_trace = baseline_correct(aligned, time_axis, baseline_window)
        ax.plot(time_axis, mean_trace, color=color, lw=2, label=f"Contrast {c}")
        ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.3)
    ax.axvline(0, color="k", linestyle="--")
    ax.set_title(fb_label)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(win)
    if col_idx == 0:
        ax.set_ylabel("ΔF/F (baseline-corrected)")
    ax.legend(fontsize=8)

fig.suptitle(f"{subject} | {region} | {fiber} | feedback_times (by contrast & feedback)", fontsize=12)
plt.tight_layout()
plt.show()
# %%























#%%
# ==============================================================================
# ==============================================================================
# ============================================================================== 
# PLOT PSTHs ACROSS SESSIONS PER NM (for each event)
# ===============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/psth_arrays"
EVENTS = ["stimOnTrigger_times", "feedback_times"]
YLIM = [-1.15, 1.55]
PERIEVENT_WINDOW = [-1, 2]
TARGET_RATE = 30  # Hz

# 🎨 Color palettes (contrast 0 → gray, others light → dark)
PALETTES = {
    "DA": sns.blend_palette(["#F7B7AE", "#EC8072", "#D44836", "#A1271A"], n_colors=4).as_hex(),
    "5-HT": sns.blend_palette(["#C5A5F0", "#A374E8", "#7F4CC8", "#572F91"], n_colors=4).as_hex(),
    "NE": sns.blend_palette(["#A0C8F5", "#64A7E9", "#2E84CC", "#166EA3"], n_colors=4).as_hex(),
    "ACh": sns.blend_palette(["#A9E2A6", "#71C769", "#45B437", "#2F7A2D"], n_colors=4).as_hex(),
}
GRAY = "#D9D9D9"

# 🧠 Updated mice per NM
MICE = {
    "DA": ["ZFM-03447", "ZFM-03448", "ZFM-03450", "ZFM-04026", "ZFM-04019", "ZFM-04022"],
    "5-HT": ["ZFM-03061", "ZFM-03065", "ZFM-03059", "ZFM-03062",
             "ZFM-04392", "ZFM-05236", "ZFM-05248", "ZFM-05245", "ZFM-05235"],
    "NE": ["ZFM-04533", "ZFM-04534", "ZFM-06271", "ZFM-06272",
           "ZFM-06171", "ZFM-06268", "ZFM-06275"],
    "ACh": ["ZFM-06305", "ZFM-06946", "ZFM-06948"]
}

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def interpolate_to_common_time(psth, current_rate, target_rate, perievent_window):
    """Interpolate PSTH array from current_rate → target_rate."""
    current_time = np.linspace(perievent_window[0], perievent_window[1], psth.shape[1])
    target_time = np.linspace(perievent_window[0], perievent_window[1],
                              int((perievent_window[1] - perievent_window[0]) * target_rate))
    interpolated = np.array([np.interp(target_time, current_time, trace) for trace in psth])
    return interpolated


def pad_to_match_length(arr, target_len):
    """Pad or trim arrays to match the target time length."""
    if arr.shape[1] == target_len:
        return arr
    elif arr.shape[1] < target_len:
        pad = target_len - arr.shape[1]
        pad_block = np.tile(arr[:, -1][:, None], (1, pad))
        return np.hstack([arr, pad_block])
    else:
        return arr[:, :target_len]


def load_mouse_psth(subject, event, target_rate=TARGET_RATE):
    """Load all .npz PSTH files for a subject, align sampling rate and length."""
    files = [f for f in os.listdir(BASE_DIR)
             if f.startswith(f"2_{subject}") and event in f and f.endswith(".npz")]
    if not files:
        print(f"⚠️ No PSTH files found for {subject} ({event})")
        return None, None

    psth_by_contrast = {}
    max_len = 0

    for file in files:
        data = np.load(os.path.join(BASE_DIR, file), allow_pickle=True)
        psth_mean = data["psth_mean"].item()
        time_axis = data["time_axis"]
        current_rate = 1 / np.median(np.diff(time_axis))

        for contrast, subdict in psth_mean.items():
            if contrast not in psth_by_contrast:
                psth_by_contrast[contrast] = {"correct": [], "incorrect": []}
            for label in ["correct", "incorrect"]:
                trace = np.array(subdict[label])
                # Interpolate if rate mismatch
                if abs(current_rate - target_rate) > 1:
                    interp_trace = interpolate_to_common_time(
                        trace[None, :], current_rate, target_rate, PERIEVENT_WINDOW)[0]
                else:
                    interp_trace = trace
                psth_by_contrast[contrast][label].append(interp_trace)
                max_len = max(max_len, len(interp_trace))

    # pad/trim to same length
    for contrast, subdict in psth_by_contrast.items():
        for label in ["correct", "incorrect"]:
            padded = [pad_to_match_length(trace[None, :], max_len)[0] for trace in subdict[label]]
            psth_by_contrast[contrast][label] = padded

    target_time = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], max_len)
    return target_time, psth_by_contrast


# =========================================================
# COMBINE ACROSS MICE
# =========================================================
def combine_across_mice(nm, event):
    combined, time_axis = {}, None
    for subject in MICE[nm]:
        t, psth_mouse = load_mouse_psth(subject, event)
        if t is None:
            continue
        time_axis = t
        for contrast, subdict in psth_mouse.items():
            if contrast not in combined:
                combined[contrast] = {"correct": [], "incorrect": []}
            for label in ["correct", "incorrect"]:
                combined[contrast][label].extend(subdict[label])

    psth_combined = {}
    for contrast, subdict in combined.items():
        psth_combined[contrast] = {}
        for label in ["correct", "incorrect"]:
            arr = np.vstack(subdict[label]) if subdict[label] else np.full((1, len(time_axis)), np.nan)
            mean = np.nanmean(arr, axis=0)
            error = sem(arr, axis=0, nan_policy="omit")
            psth_combined[contrast][label] = {"mean": mean, "sem": error}
    return time_axis, psth_combined


# =========================================================
# PLOT FUNCTION
# =========================================================
def plot_all_NMs(event):
    fig, axes = plt.subplots(4, 2, figsize=(12, 18), dpi=300, sharex=True, sharey=True)
    nms = ["DA", "5-HT", "NE", "ACh"]

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
    })

    for row, nm in enumerate(nms):
        time_axis, psth_combined = combine_across_mice(nm, event)
        if time_axis is None:
            continue

        colors = PALETTES[nm]
        contrasts = sorted(psth_combined.keys())

        for col, label in enumerate(["correct", "incorrect"]):
            ax = axes[row, col]

            # non-zero contrasts (light → dark)
            nonzero = [c for c in contrasts if c != 0]
            for i, contrast in enumerate(nonzero):
                y = psth_combined[contrast][label]["mean"]
                e = psth_combined[contrast][label]["sem"]
                color = colors[min(i, len(colors)-1)]
                ax.plot(time_axis, y, lw=3, color=color, label=f"Contrast {contrast}")
                ax.fill_between(time_axis, y - e, y + e, color=color, alpha=0.22)

            # plot gray (contrast 0) on top
            if 0 in contrasts:
                y = psth_combined[0][label]["mean"]
                e = psth_combined[0][label]["sem"]
                ax.plot(time_axis, y, lw=3.5, color=GRAY, label="Contrast 0")
                ax.fill_between(time_axis, y - e, y + e, color=GRAY, alpha=0.25)

            ax.axvline(0, color="black", linestyle="--", lw=1.8)
            ax.set_xlim(PERIEVENT_WINDOW)
            ax.set_ylim(YLIM)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Row label
            if col == 0:
                ax.set_ylabel(nm, fontsize=18, rotation=0, labelpad=25, va="center")

            # Titles
            if row == 0:
                ax.set_title("Future Correct" if label == "correct" else "Future Incorrect")

            # X-axis labels
            if row == len(nms) - 1:
                xlabel = "Time since stimulus onset (s)" if event == "stimOnTrigger_times" else "Time since feedback onset (s)"
                ax.set_xlabel(xlabel)
                ax.set_xticks(np.arange(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1] + 0.1, 0.5))
                ax.set_xticklabels([f"{x:.1f}" for x in np.arange(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1] + 0.1, 0.5)])
            else:
                ax.set_xticklabels([])

            # Legend only once
            if row == 0 and col == 1:
                ax.legend(frameon=False, loc="upper right")

    title_map = {
        "stimOnTrigger_times": "PSTHs aligned to stimulus onset — split by future outcome",
        "feedback_times": "PSTHs aligned to feedback onset — split by outcome"
    }
    plt.suptitle(title_map.get(event, event), fontsize=20, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"/home/kceniabougrova/Downloads/good_sessions_outputs/AllNMs_{event}_interp_graytop.png", dpi=300)
    plt.savefig(f"/home/kceniabougrova/Downloads/good_sessions_outputs/AllNMs_{event}_interp_graytop.pdf", dpi=300)
    plt.show()


# =========================================================
# MAIN LOOP
# =========================================================
for event in EVENTS:
    plot_all_NMs(event)


















#%%
# ==============================================================================
# ==============================================================================
# ==============================================================================
# %%
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/psth_arrays"
TRIALS_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs"  # where df_trials_*.csv live
EVENTS = ["stimOnTrigger_times", "feedback_times"]
YLIM = [-1.15, 1.55]
PERIEVENT_WINDOW = [-1, 2]
TARGET_RATE = 30  # Hz
MAX_TRIALS = 89
PROB_FILTER = 0.5

# =========================================================
# HELPER: Filter trials for neutral block
# =========================================================
def filter_neutral_trials(subject, event, eid=None):
    """Load df_trials file for subject/date/event and return indices of first 89 trials with probLeft == 0.5."""
    try:
        df_trials_file = [f for f in os.listdir(TRIALS_DIR)
                          if f.startswith("df_trials_") and subject in f and event.split("_")[0] in f and f.endswith(".csv")]
        if not df_trials_file:
            return None
        df_trials = pd.read_csv(os.path.join(TRIALS_DIR, df_trials_file[0]))
        idx = df_trials.query("probabilityLeft == @PROB_FILTER").index[:MAX_TRIALS]
        return set(idx)
    except Exception as e:
        print(f"⚠️ Could not filter trials for {subject}: {e}")
        return None


# =========================================================
# LOAD AND FILTER PSTH FILES
# =========================================================
def load_mouse_psth(subject, event, target_rate=TARGET_RATE):
    """Load .npz PSTH files for a subject and filter by first 89 neutral trials."""
    files = [f for f in os.listdir(BASE_DIR)
             if f.startswith(f"2_{subject}") and event in f and f.endswith(".npz")]
    if not files:
        print(f"⚠️ No PSTH files found for {subject} ({event})")
        return None, None

    psth_by_contrast = {}
    max_len = 0

    for file in files:
        data = np.load(os.path.join(BASE_DIR, file), allow_pickle=True)
        psth_mean = data["psth_mean"].item()
        psth_trials = data.get("psth_trials")  # optional: if stored trial-by-trial
        time_axis = data["time_axis"]
        current_rate = 1 / np.median(np.diff(time_axis))

        # 🩹 Filter trial indices (probLeft == 0.5 & first 89)
        keep_trials = filter_neutral_trials(subject, event)

        for contrast, subdict in psth_mean.items():
            if contrast not in psth_by_contrast:
                psth_by_contrast[contrast] = {"correct": [], "incorrect": []}
            for label in ["correct", "incorrect"]:
                trace = np.array(subdict[label])
                # Optionally subset if psth_trials exist
                if psth_trials is not None and keep_trials:
                    trial_idx = np.array(list(keep_trials))
                    if trial_idx.max() < psth_trials.shape[1]:
                        trace = np.nanmean(psth_trials[:, trial_idx], axis=1)

                # Interpolate if rate mismatch
                if abs(current_rate - target_rate) > 1:
                    interp_trace = np.interp(
                        np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], int((PERIEVENT_WINDOW[1]-PERIEVENT_WINDOW[0])*target_rate)),
                        np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], len(trace)),
                        trace)
                else:
                    interp_trace = trace

                psth_by_contrast[contrast][label].append(interp_trace)
                max_len = max(max_len, len(interp_trace))

    # pad/trim to same length
    for contrast, subdict in psth_by_contrast.items():
        for label in ["correct", "incorrect"]:
            padded = [np.pad(trace, (0, max_len - len(trace)), mode="edge") if len(trace) < max_len else trace[:max_len]
                      for trace in subdict[label]]
            psth_by_contrast[contrast][label] = padded

    target_time = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], max_len)
    return target_time, psth_by_contrast


# =========================================================
# PLOT FUNCTION
# =========================================================
def plot_all_NMs(event):
    fig, axes = plt.subplots(4, 2, figsize=(12, 18), dpi=300, sharex=True, sharey=True)
    nms = ["DA", "5-HT", "NE", "ACh"]

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
    })

    for row, nm in enumerate(nms):
        time_axis, psth_combined = combine_across_mice(nm, event)
        if time_axis is None:
            continue

        colors = PALETTES[nm]
        contrasts = sorted(psth_combined.keys())

        for col, label in enumerate(["correct", "incorrect"]):
            ax = axes[row, col]

            # non-zero contrasts (light → dark)
            nonzero = [c for c in contrasts if c != 0]
            for i, contrast in enumerate(nonzero):
                y = psth_combined[contrast][label]["mean"]
                e = psth_combined[contrast][label]["sem"]
                color = colors[min(i, len(colors)-1)]
                ax.plot(time_axis, y, lw=3, color=color, label=f"Contrast {contrast}", alpha=0.9)
                ax.fill_between(time_axis, y - e, y + e, color=color, alpha=0.22)

            # plot gray (contrast 0) on top
            if 0 in contrasts:
                y = psth_combined[0][label]["mean"]
                e = psth_combined[0][label]["sem"]
                ax.plot(time_axis, y, lw=3.5, color=GRAY, label="Contrast 0", alpha=0.9)
                ax.fill_between(time_axis, y - e, y + e, color=GRAY, alpha=0.25)

            ax.axvline(0, color="black", linestyle="--", lw=1.8)
            ax.set_xlim(PERIEVENT_WINDOW)
            ax.set_ylim(YLIM)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Row label
            if col == 0:
                ax.set_ylabel(nm, fontsize=18, rotation=0, labelpad=25, va="center")

            # Titles
            if row == 0:
                ax.set_title("Future Correct" if label == "correct" else "Future Incorrect")

            # X-axis labels
            if row == len(nms) - 1:
                xlabel = "Time since stimulus onset (s)" if event == "stimOnTrigger_times" else "Time since feedback onset (s)"
                ax.set_xlabel(xlabel)
                ax.set_xticks(np.arange(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1] + 0.1, 0.5))
                ax.set_xticklabels([f"{x:.1f}" for x in np.arange(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1] + 0.1, 0.5)])
            else:
                ax.set_xticklabels([])

            # Legend only once
            if row == 0 and col == 1:
                ax.legend(frameon=False, loc="upper right")

    title_map = {
        "stimOnTrigger_times": "PSTHs aligned to stimulus onset — split by future outcome",
        "feedback_times": "PSTHs aligned to feedback onset — split by outcome"
    }
    plt.suptitle(title_map.get(event, event), fontsize=20, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"/home/kceniabougrova/Downloads/good_sessions_outputs/AllNMs_{event}_interp_graytop2.png", dpi=300)
    plt.savefig(f"/home/kceniabougrova/Downloads/good_sessions_outputs/AllNMs_{event}_interp_graytop2.pdf", dpi=300)
    plt.show()


# %%
for event in EVENTS:
    plot_all_NMs(event)

#%%
# %%
# %%
# =========================================================
# COMBINE PSTHs ACROSS ALL NE MICE
# =========================================================
print("\n🧠 Combining PSTHs for NE mice...")

MICE = {
    "DA": ["ZFM-03447", "ZFM-03448", "ZFM-03450", "ZFM-04026", "ZFM-04019", "ZFM-04022"],
    "5-HT": ["ZFM-03061", "ZFM-03065", "ZFM-03059", "ZFM-03062",
             "ZFM-04392", "ZFM-05236", "ZFM-05248", "ZFM-05245", "ZFM-05235"],
    "NE": ["ZFM-04533", "ZFM-04534", "ZFM-06271", "ZFM-06272",
           "ZFM-06171", "ZFM-06268", "ZFM-06275"],
    "ACh": ["ZFM-06305", "ZFM-06946", "ZFM-06948"]
}

# Choose neuromodulator and event
NM = "NE"
EVENT = "stimOnTrigger_times"  # or "stimOnTrigger_times"

PSTH_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/psth_arrays_oldway"
TRIALS_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"

# Initialize containers
aligned_all = []
feedback_all = []
contrast_all = []

# Loop through all NE mice
for mouse in MICE[NM]:
    print(f"🐭 Loading sessions for {mouse}")
    for f in os.listdir(PSTH_DIR):
        if not f.endswith(".npy") or EVENT not in f:
            continue
        if mouse not in f:
            continue

        # Derive df_trials filename pattern
        parts = f.replace("psth_", "").replace(".npy", "").split("_")
        if len(parts) < 5:
            continue
        subject, date, region, eid, fiber = parts[0:5]

        # Find matching df_trials file
        trial_files = [t for t in os.listdir(TRIALS_DIR)
                       if t.startswith("df_trials_") and subject in t and date in t and region in t and eid in t and t.endswith(".csv")]
        if not trial_files:
            print(f"⚠️ No df_trials found for {mouse} {date}")
            continue

        df_trials = pd.read_csv(os.path.join(TRIALS_DIR, trial_files[0]))
        psth = np.load(os.path.join(PSTH_DIR, f))

        # Ensure same number of trials
        n_trials = min(psth.shape[1], df_trials.shape[0])
        psth = psth[:, :n_trials]
        df_trials = df_trials.iloc[:n_trials]

        aligned_all.append(psth)
        feedback_all.append(df_trials["feedbackType"].values)
        if "signed_contrast" in df_trials.columns:
            contrast_all.append(df_trials["signed_contrast"].values)
        else:
            contrast_all.append(np.zeros(n_trials))

# Concatenate all sessions
if aligned_all:
    psth_combined = np.concatenate(aligned_all, axis=1)
    feedback_combined = np.concatenate(feedback_all)
    contrast_combined = np.concatenate(contrast_all)
    print(f"✅ Combined shape: {psth_combined.shape}")
else:
    raise ValueError("❌ No NE PSTHs found to combine!")

# =========================================================
# COMPUTE AVERAGES
# =========================================================
PERIEVENT_WINDOW = [-1, 2]
SAMPLING_RATE = 30  # assume 30 Hz for time axis
time_vector = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], psth_combined.shape[0])

psth_good = psth_combined[:, feedback_combined == 1]
psth_error = psth_combined[:, feedback_combined == -1]

psth_good_avg = np.nanmean(psth_good, axis=1)
sem_good = sem(psth_good, axis=1, nan_policy="omit")
psth_error_avg = np.nanmean(psth_error, axis=1)
sem_error = sem(psth_error, axis=1, nan_policy="omit")

# =========================================================
# PLOT NE GROUP PSTH
# =========================================================
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

# Correct trials
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(psth_good.T, cbar=False, ax=ax1, cmap="vlag", center=0)
ax1.invert_yaxis()
ax1.axvline(x=SAMPLING_RATE, color="white", lw=3, ls="--")
ax1.set_title(f"{NM} — Correct trials")
ticks = np.linspace(0, len(time_vector) - 1, 5)
tick_labels = np.round(np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], 5), 2)
ax1.set_xticks(ticks)
ax1.set_xticklabels(tick_labels)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_vector, psth_good_avg, color="#2f9c95", lw=3)
ax2.fill_between(time_vector, psth_good_avg - sem_good, psth_good_avg + sem_good, color="#2f9c95", alpha=0.2)
ax2.axvline(0, color="black", lw=2, ls="--")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("ΔF/F (z-scored)")

# Incorrect trials
ax3 = fig.add_subplot(gs[0, 1])
sns.heatmap(psth_error.T, cbar=False, ax=ax3, cmap="vlag", center=0)
ax3.invert_yaxis()
ax3.axvline(x=SAMPLING_RATE, color="white", lw=3, ls="--")
ax3.set_title(f"{NM} — Incorrect trials")
ax3.set_xticks(ticks)
ax3.set_xticklabels(tick_labels)

ax4 = fig.add_subplot(gs[1, 1], sharey=ax2)
ax4.plot(time_vector, psth_error_avg, color="#d62828", lw=3)
ax4.fill_between(time_vector, psth_error_avg - sem_error, psth_error_avg + sem_error, color="#d62828", alpha=0.2)
ax4.axvline(0, color="black", lw=2, ls="--")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("ΔF/F (z-scored)")

fig.suptitle(f"{NM} Group PSTH — {EVENT}", y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PSTH_DIR, f"{NM}_group_{EVENT}.png"), dpi=300)
plt.show()
# %%
#%% 
# ==============================================================================
# ==============================================================================
# ==============================================================================












































































#%%
# ==============================================================================
# ==============================================================================
# ==============================================================================


df_good_BCW = pd.read_excel("/home/kceniabougrova/Downloads/good_sessions_outputs/df_good_BCW_2ndpass.xlsx") #278 sessions - removed some sessions with smaller signal


i = 58
row = df_good_BCW.iloc[i]

# Extract metadata
subject = row['subject']
date = str(row['date'])[:10]
region = row['region']
fiber = row['fiber']
eid = row['eid']
fiber = row['fiber']

print(f"📦 Session {i} — {subject} | {date} | {region} | {fiber} | {eid} {fiber}")


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
time_diffs = df_nph["times"].diff().dropna()
fs = 1 / time_diffs.median() 

array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
# print(idx_event) 


""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
df_nph.loc[idx_event,"trial_number"]=1
df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 


PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
SAMPLING_RATE = int(fs) #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR 
EVENT = "feedback_times"

sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

event_feedback = np.array(df_trials[EVENT]) #pick the feedback timestamps 

feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

psth_idx += feedback_idx



# %%
# %%
mouse=subject

psth_good = df_nph.zdFF.values[psth_idx[:,(df_trials.feedbackType == 1)]]
psth_error = df_nph.zdFF.values[psth_idx[:,(df_trials.feedbackType == -1)]]
# Calculate averages and SEM
psth_good_avg = psth_good.mean(axis=1)
sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
psth_error_avg = psth_error.mean(axis=1)
sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

time_vector = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], len(psth_good_avg))

# Create the figure and gridspec
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

# Plot the heatmap and line plot for correct trials
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
ax1.invert_yaxis()
ax1.axvline(x=SAMPLING_RATE, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
ax1.set_title('Correct Trials')
# Set x-axis tick labels to show time in seconds for the heatmaps
ticks = np.linspace(0, len(time_vector)-1, num=5)
tick_labels = np.round(np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], num=5), 2)
ax1.set_xticks(ticks)
ax1.set_xticklabels(tick_labels)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_vector, psth_good_avg, color='#2f9c95', linewidth=3) 
# ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
# ax2.fill_between(time_vector, psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
ax2.fill_between(time_vector, psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
ax2.axvline(x=0, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax2.set_ylabel('Average Value')
ax2.set_xlabel('Time (s)')

# Plot the heatmap and line plot for incorrect trials
ax3 = fig.add_subplot(gs[0, 1])
sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
ax3.invert_yaxis()
ax3.axvline(x=SAMPLING_RATE, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
ax3.set_title('Incorrect Trials')

ax3.set_xticks(ticks)
ax3.set_xticklabels(tick_labels)

ax4 = fig.add_subplot(gs[1, 1], sharey=ax2)
ax4.plot(time_vector, psth_error_avg, color='#d62828', linewidth=3)
ax4.fill_between(time_vector, psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
ax4.axvline(x=0, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax4.set_ylabel('Average Value')
ax4.set_xlabel('Time (s)')

fig.suptitle(f'calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}', y=1, fontsize=14)
plt.tight_layout()
# plt.savefig(f'/mnt/h0/kb/data/psth_npy/30082024/Fig02_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
plt.show()
# %%




#%%
# ===============================================================
# ===============================================================
# ===============================================================
# to loop over sessions to save the npy
# ===============================================================
import os
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
SAVE_DIR = os.path.join(BASE_DIR, "psth_arrays_oldway")
os.makedirs(SAVE_DIR, exist_ok=True)

PERIEVENT_WINDOW = [-1, 2]  # seconds
EVENTS = ["stimOnTrigger_times", "feedback_times"]

# =========================================================
# LOAD SESSION METADATA
# =========================================================
df_good_BCW = pd.read_excel(os.path.join(BASE_DIR, "df_good_BCW_2ndpass.xlsx"))

print(f"📋 Loaded {len(df_good_BCW)} sessions to process")

# =========================================================
# MAIN LOOP
# =========================================================
for i, row in df_good_BCW.iterrows():
    subject = row["subject"]
    date = str(row["date"])[:10]
    region = row["region"]
    fiber = row["fiber"]
    eid = row["eid"]

    print(f"\n📦 Session {i}: {subject} | {date} | {region} | {fiber}")

    # locate matching files
    df_trials_file = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith("df_trials_")
        and subject in f and date in f and region in f and eid in f
    ]
    df_nph_file = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith("df_nph_")
        and subject in f and date in f and region in f and eid in f
    ]

    if not df_trials_file or not df_nph_file:
        print("⚠️ Missing one or both files, skipping")
        continue

    df_trials_path = os.path.join(BASE_DIR, df_trials_file[0])
    df_nph_path = os.path.join(BASE_DIR, df_nph_file[0])

    # Load data
    df_trials = pd.read_csv(df_trials_path)
    df_nph = pd.read_csv(df_nph_path)

    # =====================================================
    # COMPUTE SAMPLING RATE
    # =====================================================
    time_diffs = df_nph["times"].diff().dropna()
    fs = 1 / time_diffs.median()
    fs = int(round(fs))
    print(f"   ⏱ Sampling rate: {fs} Hz")

    # photometry timestamps (in bpod clock)
    array_timestamps_bpod = np.array(df_nph["times"])

    # create trial_number column
    df_nph["trial_number"] = 0
    idx_event = np.searchsorted(array_timestamps_bpod, np.array(df_trials["intervals_0"]))
    df_nph.loc[idx_event, "trial_number"] = 1
    df_nph["trial_number"] = df_nph["trial_number"].cumsum()

    # =====================================================
    # CREATE PSTH INDEX MATRICES
    # =====================================================
    sample_window = np.arange(PERIEVENT_WINDOW[0] * fs, PERIEVENT_WINDOW[1] * fs + 1)
    n_trials = df_trials.shape[0]

    for EVENT in EVENTS:
        if EVENT not in df_trials.columns:
            print(f"   ⚠️ Missing column {EVENT}, skipping")
            continue

        event_times = np.array(df_trials[EVENT].dropna())
        if len(event_times) == 0:
            print(f"   ⚠️ No {EVENT} timestamps found")
            continue

        feedback_idx = np.searchsorted(array_timestamps_bpod, event_times)
        psth_idx = np.tile(sample_window[:, np.newaxis], (1, len(feedback_idx))) + feedback_idx

        # =====================================================
        # SAVE PSTH IDX ARRAY
        # =====================================================
        save_name = f"psth_{subject}_{date}_{region}_{eid}_{fiber}_{EVENT}.npy"
        save_path = os.path.join(SAVE_DIR, save_name)
        np.save(save_path, psth_idx)
        print(f"   💾 Saved → {save_path}")

print("\n✅ All done! PSTH index arrays created and saved.")

# %%
# ===============================================================
# ===============================================================
# ===============================================================
# each session plot for stimOn and feedback times, saved 




# %%
# %%
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
PSTH_SAVE_DIR = os.path.join(BASE_DIR, "psth_arrays_oldway")
os.makedirs(PSTH_SAVE_DIR, exist_ok=True)

df_good_BCW = pd.read_excel(os.path.join(BASE_DIR, "df_good_BCW_2ndpass.xlsx"))

EVENTS = ["stimOnTrigger_times", "feedback_times"]
PERIEVENT_WINDOW = [-1, 2]  # seconds
SAVE = True  # change to False to only plot

# =========================================================
# FUNCTION TO FIND MATCHING FILES (ignoring numeric prefix)
# =========================================================
def find_matching_file(base_dir, prefix, subject, date, region, eid, ext=".csv"):
    for f in os.listdir(base_dir):
        if not f.startswith(prefix) or not f.endswith(ext):
            continue
        if re.search(rf"{subject}.*{date}.*{region}.*{eid}", f):
            return os.path.join(base_dir, f)
    return None


# =========================================================
# MAIN LOOP THROUGH SESSIONS
# =========================================================
for i, row in df_good_BCW.iterrows():
    subject = row["subject"]
    date = str(row["date"])[:10]
    region = str(row["region"])
    fiber = str(row["fiber"])
    eid = str(row["eid"])
    nm = str(row["NM"])

    print(f"\n📦 Session {i}: {subject} | {date} | {region} | {fiber} | {nm}")

    df_trials_path = find_matching_file(BASE_DIR, "df_trials_", subject, date, region, eid)
    df_nph_path = find_matching_file(BASE_DIR, "df_nph_", subject, date, region, eid)

    if not df_trials_path or not df_nph_path:
        print("⚠️ Missing df_trials or df_nph file, skipping session")
        continue

    # Load files
    df_trials = pd.read_csv(df_trials_path)
    df_nph = pd.read_csv(df_nph_path)

    # Compute sampling rate
    time_diffs = df_nph["times"].diff().dropna()
    fs_measured = 1 / time_diffs.median()
    SAMPLING_RATE = 15 if abs(fs_measured - 15) < abs(fs_measured - 30) else 30
    print(f"📊 Sampling rate detected: {SAMPLING_RATE} Hz (measured ≈ {fs_measured:.2f} Hz)")

    # Build sample window (−1 s → +2 s)
    sample_window = np.arange(
        int(PERIEVENT_WINDOW[0] * SAMPLING_RATE),
        int(PERIEVENT_WINDOW[1] * SAMPLING_RATE) + 1,
    )
    n_trials = df_trials.shape[0]
    psth_idx = np.tile(sample_window[:, np.newaxis], (1, n_trials))

    array_timestamps = df_nph["times"].values

    for EVENT in EVENTS:
        print(f"   🧩 Aligning to: {EVENT}")
        event_times = np.array(df_trials[EVENT].dropna())
        event_idx = np.searchsorted(array_timestamps, event_times)

        # Align window indices to events
        psth_idx_event = psth_idx.copy()
        psth_idx_event += event_idx
        psth_idx_event = psth_idx_event.clip(0, len(df_nph) - 1)

        # Extract ΔF/F signal
        signal = df_nph["zdFF"].values
        aligned_signal = signal[psth_idx_event]

        # Save PSTH array
        save_name = f"psth_{subject}_{date}_{region}_{eid}_{fiber}_{EVENT}.npy"
        save_path = os.path.join(PSTH_SAVE_DIR, save_name)
        np.save(save_path, aligned_signal)
        print(f"💾 Saved PSTH → {save_path}")

        # ----------------------------------------------------
        # Plot heatmap and average trace
        # ----------------------------------------------------
        feedback_type = df_trials["feedbackType"].fillna(0).astype(int)
        psth_good = aligned_signal[:, feedback_type == 1]
        psth_error = aligned_signal[:, feedback_type == -1]

        psth_good_avg = np.nanmean(psth_good, axis=1)
        sem_good = sem(psth_good, axis=1, nan_policy="omit")
        psth_error_avg = np.nanmean(psth_error, axis=1)
        sem_error = sem(psth_error, axis=1, nan_policy="omit")

        time_vector = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], len(psth_good_avg))

        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

        # Correct trials
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(psth_good.T, cbar=False, ax=ax1, cmap="vlag", center=0)
        ax1.invert_yaxis()
        ax1.axvline(x=SAMPLING_RATE, color="white", lw=3, ls="--")
        ax1.set_title("Correct trials")

        ticks = np.linspace(0, len(time_vector) - 1, 5)
        tick_labels = np.round(np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], 5), 2)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(tick_labels)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time_vector, psth_good_avg, color="#2f9c95", lw=3)
        ax2.fill_between(time_vector, psth_good_avg - sem_good, psth_good_avg + sem_good, color="#2f9c95", alpha=0.2)
        ax2.axvline(0, color="black", lw=2, ls="--")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("ΔF/F (z-scored)")

        # Incorrect trials
        ax3 = fig.add_subplot(gs[0, 1])
        sns.heatmap(psth_error.T, cbar=False, ax=ax3, cmap="vlag", center=0)
        ax3.invert_yaxis()
        ax3.axvline(x=SAMPLING_RATE, color="white", lw=3, ls="--")
        ax3.set_title("Incorrect trials")
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(tick_labels)

        ax4 = fig.add_subplot(gs[1, 1], sharey=ax2)
        ax4.plot(time_vector, psth_error_avg, color="#d62828", lw=3)
        ax4.fill_between(time_vector, psth_error_avg - sem_error, psth_error_avg + sem_error, color="#d62828", alpha=0.2)
        ax4.axvline(0, color="black", lw=2, ls="--")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("ΔF/F (z-scored)")

        fig.suptitle(f"{nm} | {subject} | {date} | {region} | {fiber} | {EVENT}", y=1.02, fontsize=14)
        plt.tight_layout()

        if SAVE:
            out_name = f"heatmap_{subject}_{date}_{region}_{eid}_{fiber}_{EVENT}.png"
            out_path = os.path.join(PSTH_SAVE_DIR, out_name)
            plt.savefig(out_path, dpi=300)
            print(f"🖼️  Saved plot → {out_path}")
            plt.close()
        else:
            plt.show()

print("\n✅ Done generating PSTHs for all sessions!")
# %%

# ========================================================
# ========================================================
# ========================================================
# cool, no baseline correction here
# %%
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
PSTH_DIR = os.path.join(BASE_DIR, "psth_arrays_oldway")
TRIALS_DIR = BASE_DIR

EVENTS = ["stimOnTrigger_times", "feedback_times"]
PERIEVENT_WINDOW = [-1, 2]
TARGET_FR = 30
target_duration = int((PERIEVENT_WINDOW[1] - PERIEVENT_WINDOW[0]) * TARGET_FR)

MICE = {
    "DA": ["ZFM-03447", "ZFM-03448", "ZFM-03450", "ZFM-04026", "ZFM-04019", "ZFM-04022"],
    "5-HT": ["ZFM-03061", "ZFM-03065", "ZFM-03059", "ZFM-03062",
             "ZFM-04392", "ZFM-05236", "ZFM-05248", "ZFM-05245", "ZFM-05235"],
    "NE": ["ZFM-04533", "ZFM-04534", "ZFM-06271", "ZFM-06272",
           "ZFM-06171", "ZFM-06268", "ZFM-06275"],
    "ACh": ["ZFM-06305", "ZFM-06946", "ZFM-06948"]
}

PALETTES = {
    "DA": sns.blend_palette(["#F7B7AE", "#EC8072", "#D44836", "#A1271A"], n_colors=4).as_hex(),
    "5-HT": sns.blend_palette(["#C5A5F0", "#A374E8", "#7F4CC8", "#572F91"], n_colors=4).as_hex(),
    "NE": sns.blend_palette(["#A0C8F5", "#64A7E9", "#2E84CC", "#166EA3"], n_colors=4).as_hex(),
    "ACh": sns.blend_palette(["#A9E2A6", "#71C769", "#45B437", "#2F7A2D"], n_colors=4).as_hex(),
}
GRAY = "#D9D9D9"

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize_photometry_segment(segment, target_duration):
    """Interpolate 1D PSTH to a common sample length."""
    orig_len = len(segment)
    orig_t = np.linspace(0, orig_len - 1, orig_len)
    target_t = np.linspace(0, orig_len - 1, target_duration)
    return np.interp(target_t, orig_t, segment)

def normalize_psth_matrix(psth_matrix, target_duration):
    """Normalize 2D PSTH (time × trials)."""
    normalized = np.zeros((target_duration, psth_matrix.shape[1]))
    for i in range(psth_matrix.shape[1]):
        normalized[:, i] = normalize_photometry_segment(psth_matrix[:, i], target_duration)
    return normalized

# =========================================================
# MAIN LOOP THROUGH NEUROMODULATORS AND EVENTS
# =========================================================
for NM, mice_list in MICE.items():
    print(f"\n🧠 Processing {NM}...")
    colors_nm = [GRAY] + PALETTES[NM]

    for EVENT in EVENTS:
        print(f"   ⚙️ Event: {EVENT}")

        aligned_all, feedback_all, contrast_all = [], [], []

        # -------------------------------------------------
        # Load all sessions for this NM
        # -------------------------------------------------
        for mouse in mice_list:
            for f in os.listdir(PSTH_DIR):
                if not f.endswith(".npy") or EVENT not in f or mouse not in f:
                    continue

                parts = f.replace("psth_", "").replace(".npy", "").split("_")
                if len(parts) < 5:
                    continue
                subject, date, region, eid, fiber = parts[:5]

                trial_files = [
                    t for t in os.listdir(TRIALS_DIR)
                    if t.startswith("df_trials_")
                    and subject in t and date in t and region in t and eid in t and t.endswith(".csv")
                ]
                if not trial_files:
                    continue

                df_trials = pd.read_csv(os.path.join(TRIALS_DIR, trial_files[0]))
                psth = np.load(os.path.join(PSTH_DIR, f))

                # detect session FR and normalize
                session_dur = PERIEVENT_WINDOW[1] - PERIEVENT_WINDOW[0]
                fs_session = round(psth.shape[0] / session_dur)
                psth = normalize_psth_matrix(psth, target_duration)

                # truncate to same trial count
                n_trials = min(psth.shape[1], df_trials.shape[0])
                psth = psth[:, :n_trials]
                df_trials = df_trials.iloc[:n_trials]

                aligned_all.append(psth)
                feedback_all.append(df_trials["feedbackType"].values)
                if "allContrasts" in df_trials.columns:
                    contrast_all.append(df_trials["allContrasts"].values)
                elif "signed_contrast" in df_trials.columns:
                    contrast_all.append(np.abs(df_trials["signed_contrast"].values))
                else:
                    contrast_all.append(np.zeros(n_trials))

        # -------------------------------------------------
        # Combine across sessions
        # -------------------------------------------------
        if not aligned_all:
            print(f"⚠️ No sessions found for {NM} ({EVENT})")
            continue

        psth_combined = np.concatenate(aligned_all, axis=1)
        feedback_combined = np.concatenate(feedback_all)
        contrast_combined = np.concatenate(contrast_all)

        contrast_combined = np.round(contrast_combined, 4)
        contrasts = np.sort(np.unique(contrast_combined))
        print(f"   🎨 Contrasts found: {contrasts}")

        # -------------------------------------------------
        # Compute means per contrast
        # -------------------------------------------------
        mean_correct, sem_correct, mean_incorrect, sem_incorrect = {}, {}, {}, {}
        for c in contrasts:
            idx_c = np.isclose(contrast_combined, c, atol=1e-4)
            idx_correct = (feedback_combined == 1) & idx_c
            idx_incorrect = (feedback_combined == -1) & idx_c

            psth_c_correct = psth_combined[:, idx_correct]
            psth_c_incorrect = psth_combined[:, idx_incorrect]

            mean_correct[c] = np.nanmean(psth_c_correct, axis=1)
            sem_correct[c] = sem(psth_c_correct, axis=1, nan_policy="omit")
            mean_incorrect[c] = np.nanmean(psth_c_incorrect, axis=1)
            sem_incorrect[c] = sem(psth_c_incorrect, axis=1, nan_policy="omit")

        # -------------------------------------------------
        # Plot group PSTH split by contrast
        # -------------------------------------------------
        time_vector = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], psth_combined.shape[0])
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        for i, (label, mean_dict, sem_dict) in enumerate(zip(
                ["Correct trials", "Incorrect trials"],
                [mean_correct, mean_incorrect],
                [sem_correct, sem_incorrect])):

            ax = axes[i]
            for j, c in enumerate(contrasts):
                color = colors_nm[j] if j < len(colors_nm) else colors_nm[-1]
                ax.plot(time_vector, mean_dict[c], lw=3, color=color, label=f"Contrast {c:.4f}".rstrip("0").rstrip("."))
                ax.fill_between(time_vector,
                                mean_dict[c] - sem_dict[c],
                                mean_dict[c] + sem_dict[c],
                                color=color, alpha=0.25)

            ax.axvline(0, color="black", lw=2, ls="--")
            ax.set_xlim(PERIEVENT_WINDOW)
            ax.set_xlabel("Time (s)")
            if i == 0:
                ax.set_ylabel("ΔF/F (z-scored)")
            ax.set_title(label)
            ax.legend(frameon=False, fontsize=9)

        fig.suptitle(f"{NM} — {EVENT} (normalized to {TARGET_FR} Hz)", y=1.03, fontsize=15)
        plt.tight_layout()
        save_path = os.path.join(PSTH_DIR, f"{NM}_group_{EVENT}_byContrast_norm.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   🖼️ Saved → {save_path}")

print("\n✅ Done generating group PSTHs for all neuromodulators and events.")
# %%

# ===============================================================
# ===============================================================
# ===============================================================
# baseline correction here

# %%
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
PSTH_DIR = os.path.join(BASE_DIR, "psth_arrays_oldway")
TRIALS_DIR = BASE_DIR

EVENTS = ["stimOnTrigger_times", "feedback_times"]
PERIEVENT_WINDOW = [-1, 2]
TARGET_FR = 30
target_duration = int((PERIEVENT_WINDOW[1] - PERIEVENT_WINDOW[0]) * TARGET_FR)
BASELINE_WINDOW = [-0.05, 0]  # seconds for baseline correction

MICE = {
    "DA": ["ZFM-03447", "ZFM-03448", "ZFM-03450", "ZFM-04026", "ZFM-04019", "ZFM-04022"],
    "5-HT": ["ZFM-03061", "ZFM-03065", "ZFM-03059", "ZFM-03062",
             "ZFM-04392", "ZFM-05236", "ZFM-05248", "ZFM-05245", "ZFM-05235"],
    "NE": ["ZFM-04533", "ZFM-04534", "ZFM-06271", "ZFM-06272",
           "ZFM-06171", "ZFM-06268", "ZFM-06275"],
    "ACh": ["ZFM-06305", "ZFM-06946", "ZFM-06948"]
}

PALETTES = {
    "DA": sns.blend_palette(["#F7B7AE", "#EC8072", "#D44836", "#A1271A"], n_colors=4).as_hex(),
    "5-HT": sns.blend_palette(["#C5A5F0", "#A374E8", "#7F4CC8", "#572F91"], n_colors=4).as_hex(),
    "NE": sns.blend_palette(["#A0C8F5", "#64A7E9", "#2E84CC", "#166EA3"], n_colors=4).as_hex(),
    "ACh": sns.blend_palette(["#A9E2A6", "#71C769", "#45B437", "#2F7A2D"], n_colors=4).as_hex(),
}
GRAY = "#D9D9D9"

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize_photometry_segment(segment, target_duration):
    """Interpolate 1D PSTH to a common sample length."""
    orig_len = len(segment)
    orig_t = np.linspace(0, orig_len - 1, orig_len)
    target_t = np.linspace(0, orig_len - 1, target_duration)
    return np.interp(target_t, orig_t, segment)

def normalize_psth_matrix(psth_matrix, target_duration):
    """Normalize 2D PSTH (time × trials)."""
    normalized = np.zeros((target_duration, psth_matrix.shape[1]))
    for i in range(psth_matrix.shape[1]):
        normalized[:, i] = normalize_photometry_segment(psth_matrix[:, i], target_duration)
    return normalized

def baseline_correct_per_trial(psth_matrix, time_vector, baseline_window):
    """Subtract per-trial baseline mean (between baseline_window[0] and [1])."""
    baseline_mask = (time_vector >= baseline_window[0]) & (time_vector <= baseline_window[1])
    baselines = np.nanmean(psth_matrix[baseline_mask, :], axis=0, keepdims=True)
    psth_corrected = psth_matrix - baselines
    return psth_corrected

# =========================================================
# MAIN LOOP THROUGH NEUROMODULATORS AND EVENTS
# =========================================================
for NM, mice_list in MICE.items():
    print(f"\n🧠 Processing {NM}...")
    colors_nm = [GRAY] + PALETTES[NM]

    for EVENT in EVENTS:
        print(f"   ⚙️ Event: {EVENT}")

        aligned_all, feedback_all, contrast_all = [], [], []

        # -------------------------------------------------
        # Load all sessions for this NM
        # -------------------------------------------------
        for mouse in mice_list:
            for f in os.listdir(PSTH_DIR):
                if not f.endswith(".npy") or EVENT not in f or mouse not in f:
                    continue

                parts = f.replace("psth_", "").replace(".npy", "").split("_")
                if len(parts) < 5:
                    continue
                subject, date, region, eid, fiber = parts[:5]

                trial_files = [
                    t for t in os.listdir(TRIALS_DIR)
                    if t.startswith("df_trials_")
                    and subject in t and date in t and region in t and eid in t and t.endswith(".csv")
                ]
                if not trial_files:
                    continue

                df_trials = pd.read_csv(os.path.join(TRIALS_DIR, trial_files[0]))
                psth = np.load(os.path.join(PSTH_DIR, f))

                # detect session FR and normalize
                session_dur = PERIEVENT_WINDOW[1] - PERIEVENT_WINDOW[0]
                fs_session = round(psth.shape[0] / session_dur)
                psth = normalize_psth_matrix(psth, target_duration)

                # truncate to same trial count
                n_trials = min(psth.shape[1], df_trials.shape[0])
                psth = psth[:, :n_trials]
                df_trials = df_trials.iloc[:n_trials]

                aligned_all.append(psth)
                feedback_all.append(df_trials["feedbackType"].values)
                if "allContrasts" in df_trials.columns:
                    contrast_all.append(df_trials["allContrasts"].values)
                elif "signed_contrast" in df_trials.columns:
                    contrast_all.append(np.abs(df_trials["signed_contrast"].values))
                else:
                    contrast_all.append(np.zeros(n_trials))

        # -------------------------------------------------
        # Combine across sessions
        # -------------------------------------------------
        if not aligned_all:
            print(f"⚠️ No sessions found for {NM} ({EVENT})")
            continue

        psth_combined = np.concatenate(aligned_all, axis=1)
        feedback_combined = np.concatenate(feedback_all)
        contrast_combined = np.concatenate(contrast_all)

        contrast_combined = np.round(contrast_combined, 4)
        contrasts = np.sort(np.unique(contrast_combined))
        print(f"   🎨 Contrasts found: {contrasts}")

        # -------------------------------------------------
        # Baseline correction (per trial)
        # -------------------------------------------------
        time_vector = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], psth_combined.shape[0])
        psth_combined = baseline_correct_per_trial(psth_combined, time_vector, BASELINE_WINDOW)

        # -------------------------------------------------
        # Compute means per contrast
        # -------------------------------------------------
        mean_correct, sem_correct, mean_incorrect, sem_incorrect = {}, {}, {}, {}
        for c in contrasts:
            idx_c = np.isclose(contrast_combined, c, atol=1e-4)
            idx_correct = (feedback_combined == 1) & idx_c
            idx_incorrect = (feedback_combined == -1) & idx_c

            psth_c_correct = psth_combined[:, idx_correct]
            psth_c_incorrect = psth_combined[:, idx_incorrect]

            mean_correct[c] = np.nanmean(psth_c_correct, axis=1)
            sem_correct[c] = sem(psth_c_correct, axis=1, nan_policy="omit")
            mean_incorrect[c] = np.nanmean(psth_c_incorrect, axis=1)
            sem_incorrect[c] = sem(psth_c_incorrect, axis=1, nan_policy="omit")

        # -------------------------------------------------
        # Plot group PSTH split by contrast
        # -------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        for i, (label, mean_dict, sem_dict) in enumerate(zip(
                ["Correct trials", "Incorrect trials"],
                [mean_correct, mean_incorrect],
                [sem_correct, sem_incorrect])):

            ax = axes[i]
            for j, c in enumerate(contrasts):
                color = colors_nm[j] if j < len(colors_nm) else colors_nm[-1]
                ax.plot(time_vector, mean_dict[c], lw=3, color=color, label=f"Contrast {c:.4f}".rstrip("0").rstrip("."))
                ax.fill_between(time_vector,
                                mean_dict[c] - sem_dict[c],
                                mean_dict[c] + sem_dict[c],
                                color=color, alpha=0.25)

            ax.axvline(0, color="black", lw=2, ls="--")
            ax.set_xlim(PERIEVENT_WINDOW)
            ax.set_xlabel("Time (s)")
            if i == 0:
                ax.set_ylabel("ΔF/F (z-scored, baseline-corrected)")
            ax.set_title(label)
            ax.legend(frameon=False, fontsize=9)

        fig.suptitle(f"{NM} — {EVENT} (baseline aligned −0.05→0 s, norm {TARGET_FR} Hz)", y=1.03, fontsize=15)
        plt.tight_layout()
        save_path = os.path.join(PSTH_DIR, f"{NM}_group_{EVENT}_byContrast_norm_baselinecorr.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   🖼️ Saved → {save_path}")

print("\n✅ Done generating baseline-corrected group PSTHs for all neuromodulators and events.")
# %%
""" 
PERFECT
"""
# %%
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
PSTH_DIR = os.path.join(BASE_DIR, "psth_arrays_oldway")
TRIALS_DIR = BASE_DIR

EVENTS = ["stimOnTrigger_times", "feedback_times"]
PERIEVENT_WINDOW = [-1, 2]
TARGET_FR = 30
target_duration = int((PERIEVENT_WINDOW[1] - PERIEVENT_WINDOW[0]) * TARGET_FR)
BASELINE_WINDOW = [-0.05, 0]  # seconds for baseline correction

MICE = {
    "DA": ["ZFM-03447", "ZFM-03448", "ZFM-03450", "ZFM-04026", "ZFM-04019", "ZFM-04022"],
    "5-HT": ["ZFM-03061", "ZFM-03065", "ZFM-03059", "ZFM-03062",
             "ZFM-04392", "ZFM-05236", "ZFM-05248", "ZFM-05245", "ZFM-05235"],
    "NE": ["ZFM-04533", "ZFM-04534", "ZFM-06271", "ZFM-06272",
           "ZFM-06171", "ZFM-06268", "ZFM-06275"],
    "ACh": ["ZFM-06305", "ZFM-06946", "ZFM-06948"]
}

PALETTES = {
    "DA": sns.blend_palette(["#F7B7AE", "#EC8072", "#D44836", "#A1271A"], n_colors=4).as_hex(),
    "5-HT": sns.blend_palette(["#C5A5F0", "#A374E8", "#7F4CC8", "#572F91"], n_colors=4).as_hex(),
    "NE": sns.blend_palette(["#A0C8F5", "#64A7E9", "#2E84CC", "#166EA3"], n_colors=4).as_hex(),
    "ACh": sns.blend_palette(["#A9E2A6", "#71C769", "#45B437", "#2F7A2D"], n_colors=4).as_hex(),
}
GRAY = "#D9D9D9"

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize_photometry_segment(segment, target_duration):
    """Interpolate 1D PSTH to a common sample length."""
    orig_len = len(segment)
    orig_t = np.linspace(0, orig_len - 1, orig_len)
    target_t = np.linspace(0, orig_len - 1, target_duration)
    return np.interp(target_t, orig_t, segment)

def normalize_psth_matrix(psth_matrix, target_duration):
    """Normalize 2D PSTH (time × trials)."""
    normalized = np.zeros((target_duration, psth_matrix.shape[1]))
    for i in range(psth_matrix.shape[1]):
        normalized[:, i] = normalize_photometry_segment(psth_matrix[:, i], target_duration)
    return normalized

def baseline_correct_per_trial(psth_matrix, time_vector, baseline_window):
    """Subtract per-trial baseline mean (between baseline_window[0] and [1])."""
    baseline_mask = (time_vector >= baseline_window[0]) & (time_vector <= baseline_window[1])
    baselines = np.nanmean(psth_matrix[baseline_mask, :], axis=0, keepdims=True)
    psth_corrected = psth_matrix - baselines
    return psth_corrected

# =========================================================
# MAIN LOOP THROUGH NEUROMODULATORS AND EVENTS
# =========================================================
for NM, mice_list in MICE.items():
    print(f"\n🧠 Processing {NM}...")
    colors_nm = [GRAY] + PALETTES[NM]

    for EVENT in EVENTS:
        print(f"   ⚙️ Event: {EVENT}")

        aligned_all, feedback_all, contrast_all = [], [], []

        # -------------------------------------------------
        # Load all sessions for this NM
        # -------------------------------------------------
        for mouse in mice_list:
            for f in os.listdir(PSTH_DIR):
                if not f.endswith(".npy") or EVENT not in f or mouse not in f:
                    continue

                parts = f.replace("psth_", "").replace(".npy", "").split("_")
                if len(parts) < 5:
                    continue
                subject, date, region, eid, fiber = parts[:5]

                trial_files = [
                    t for t in os.listdir(TRIALS_DIR)
                    if t.startswith("df_trials_")
                    and subject in t and date in t and region in t and eid in t and t.endswith(".csv")
                ]
                if not trial_files:
                    continue

                df_trials = pd.read_csv(os.path.join(TRIALS_DIR, trial_files[0]))
                psth = np.load(os.path.join(PSTH_DIR, f))

                # detect session FR and normalize
                session_dur = PERIEVENT_WINDOW[1] - PERIEVENT_WINDOW[0]
                fs_session = round(psth.shape[0] / session_dur)
                psth = normalize_psth_matrix(psth, target_duration)

                # truncate to same trial count
                n_trials = min(psth.shape[1], df_trials.shape[0])
                psth = psth[:, :n_trials]
                df_trials = df_trials.iloc[:n_trials]

                aligned_all.append(psth)
                feedback_all.append(df_trials["feedbackType"].values)
                if "allContrasts" in df_trials.columns:
                    contrast_all.append(df_trials["allContrasts"].values)
                elif "signed_contrast" in df_trials.columns:
                    contrast_all.append(np.abs(df_trials["signed_contrast"].values))
                else:
                    contrast_all.append(np.zeros(n_trials))

        # -------------------------------------------------
        # Combine across sessions
        # -------------------------------------------------
        if not aligned_all:
            print(f"⚠️ No sessions found for {NM} ({EVENT})")
            continue

        psth_combined = np.concatenate(aligned_all, axis=1)
        feedback_combined = np.concatenate(feedback_all)
        contrast_combined = np.concatenate(contrast_all)

        contrast_combined = np.round(contrast_combined, 4)
        contrasts = np.sort(np.unique(contrast_combined))
        print(f"   🎨 Contrasts found: {contrasts}")

        # -------------------------------------------------
        # Baseline correction (per trial)
        # -------------------------------------------------
        time_vector = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], psth_combined.shape[0])
        psth_combined = baseline_correct_per_trial(psth_combined, time_vector, BASELINE_WINDOW)

        # -------------------------------------------------
        # Compute means per contrast
        # -------------------------------------------------
        mean_correct, sem_correct, mean_incorrect, sem_incorrect = {}, {}, {}, {}
        n_correct_total, n_incorrect_total = 0, 0

        for c in contrasts:
            idx_c = np.isclose(contrast_combined, c, atol=1e-4)
            idx_correct = (feedback_combined == 1) & idx_c
            idx_incorrect = (feedback_combined == -1) & idx_c

            psth_c_correct = psth_combined[:, idx_correct]
            psth_c_incorrect = psth_combined[:, idx_incorrect]

            mean_correct[c] = np.nanmean(psth_c_correct, axis=1)
            sem_correct[c] = sem(psth_c_correct, axis=1, nan_policy="omit")
            mean_incorrect[c] = np.nanmean(psth_c_incorrect, axis=1)
            sem_incorrect[c] = sem(psth_c_incorrect, axis=1, nan_policy="omit")

            n_correct_total += np.sum(idx_correct)
            n_incorrect_total += np.sum(idx_incorrect)

        # -------------------------------------------------
        # Plot group PSTH split by contrast
        # -------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        for i, (label, mean_dict, sem_dict, n_trials_total) in enumerate(zip(
                ["Correct trials", "Incorrect trials"],
                [mean_correct, mean_incorrect],
                [sem_correct, sem_incorrect],
                [n_correct_total, n_incorrect_total])):

            ax = axes[i]
            for j, c in enumerate(contrasts):
                color = colors_nm[j] if j < len(colors_nm) else colors_nm[-1]
                ax.plot(time_vector, mean_dict[c], lw=3, color=color, label=f"Contrast {c:.4f}".rstrip("0").rstrip("."))
                ax.fill_between(time_vector,
                                mean_dict[c] - sem_dict[c],
                                mean_dict[c] + sem_dict[c],
                                color=color, alpha=0.25)

            ax.axvline(0, color="black", lw=2, ls="--")
            ax.set_xlim(PERIEVENT_WINDOW)
            ax.set_xlabel("Time (s)")
            if i == 0:
                # ax.set_ylabel("ΔF/F (z-scored, baseline-corrected)")
                ax.set_ylabel("neuromodulator activity")
            ax.set_title(label)

            # remove frame lines (top & right)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)


            # ➕ Add number of plotted trials
            ax.text(0.02, 0.95, f"n={n_trials_total}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=10, fontweight="bold", color="gray")

            # only show legend on the right subplot
            if i == 1:
                ax.legend(frameon=False, fontsize=9, loc="upper right")


        fig.suptitle(f"{NM} — {EVENT} (baseline −0.05→0 s, norm {TARGET_FR} Hz)", y=1.03, fontsize=15)
        plt.tight_layout()

        # Save both PNG and PDF
        save_png = os.path.join(PSTH_DIR, f"{NM}_group_{EVENT}_byContrast_norm_baselinecorr.png")
        save_pdf = os.path.join(PSTH_DIR, f"{NM}_group_{EVENT}_byContrast_norm_baselinecorr.pdf")
        plt.savefig(save_png, dpi=300)
        plt.savefig(save_pdf, dpi=300)
        plt.close()
        print(f"   🖼️ Saved → {save_png}")
        print(f"   📄 Saved → {save_pdf}")

print("\n✅ Done generating baseline-corrected group PSTHs for all neuromodulators and events.")
# %%

# %%
#-1.25 to 1.45




#%%
# ======================================================
# ======================================================
# ======================================================
# now all together as subplots in a major plot, per event 
# %%
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
PSTH_DIR = os.path.join(BASE_DIR, "psth_arrays_oldway")
TRIALS_DIR = BASE_DIR

EVENTS = ["stimOnTrigger_times", "feedback_times"]
PERIEVENT_WINDOW = [-1, 2]
TARGET_FR = 30
target_duration = int((PERIEVENT_WINDOW[1] - PERIEVENT_WINDOW[0]) * TARGET_FR)
BASELINE_WINDOW = [-0.05, 0]
YLIM = [-1.25, 1.45]

MICE = {
    "DA": ["ZFM-03447", "ZFM-03448", "ZFM-03450", "ZFM-04026", "ZFM-04019", "ZFM-04022"],
    "5-HT": ["ZFM-03061", "ZFM-03065", "ZFM-03059", "ZFM-03062",
             "ZFM-04392", "ZFM-05236", "ZFM-05248", "ZFM-05245", "ZFM-05235"],
    "NE": ["ZFM-04533", "ZFM-04534", "ZFM-06271", "ZFM-06272",
           "ZFM-06171", "ZFM-06268", "ZFM-06275"],
    "ACh": ["ZFM-06305", "ZFM-06946", "ZFM-06948"]
}

PALETTES = {
    "DA": sns.blend_palette(["#F7B7AE", "#EC8072", "#D44836", "#A1271A"], n_colors=4).as_hex(),
    "5-HT": sns.blend_palette(["#C5A5F0", "#A374E8", "#7F4CC8", "#572F91"], n_colors=4).as_hex(),
    "NE": sns.blend_palette(["#A0C8F5", "#64A7E9", "#2E84CC", "#166EA3"], n_colors=4).as_hex(),
    "ACh": sns.blend_palette(["#A9E2A6", "#71C769", "#45B437", "#2F7A2D"], n_colors=4).as_hex(),
}
GRAY = "#D9D9D9"

# =========================================================
# Helper functions
# =========================================================
def normalize_photometry_segment(segment, target_duration):
    orig_len = len(segment)
    orig_t = np.linspace(0, orig_len - 1, orig_len)
    target_t = np.linspace(0, orig_len - 1, target_duration)
    return np.interp(target_t, orig_t, segment)

def normalize_psth_matrix(psth_matrix, target_duration):
    normalized = np.zeros((target_duration, psth_matrix.shape[1]))
    for i in range(psth_matrix.shape[1]):
        normalized[:, i] = normalize_photometry_segment(psth_matrix[:, i], target_duration)
    return normalized

def baseline_correct_per_trial(psth_matrix, time_vector, baseline_window):
    baseline_mask = (time_vector >= baseline_window[0]) & (time_vector <= baseline_window[1])
    baselines = np.nanmean(psth_matrix[baseline_mask, :], axis=0, keepdims=True)
    return psth_matrix - baselines

# =========================================================
# Main plotting loop: one figure per EVENT
# =========================================================
for EVENT in EVENTS:
    print(f"\n🧩 Generating summary grid for {EVENT}")

    fig, axes = plt.subplots(4, 2, figsize=(9, 16), sharex=True, sharey=True)
    time_vector = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], target_duration)

    for row_idx, NM in enumerate(["DA", "5-HT", "NE", "ACh"]):
        colors_nm = [GRAY] + PALETTES[NM]
        aligned_all, feedback_all, contrast_all = [], [], []

        # Load and combine sessions for each NM
        for mouse in MICE[NM]:
            for f in os.listdir(PSTH_DIR):
                if not f.endswith(".npy") or EVENT not in f or mouse not in f:
                    continue
                parts = f.replace("psth_", "").replace(".npy", "").split("_")
                if len(parts) < 5:
                    continue
                subject, date, region, eid, fiber = parts[:5]
                trial_files = [t for t in os.listdir(TRIALS_DIR)
                               if t.startswith("df_trials_") and subject in t and date in t and region in t and eid in t and t.endswith(".csv")]
                if not trial_files:
                    continue
                df_trials = pd.read_csv(os.path.join(TRIALS_DIR, trial_files[0]))
                psth = np.load(os.path.join(PSTH_DIR, f))

                # normalize
                fs_session = round(psth.shape[0] / (PERIEVENT_WINDOW[1] - PERIEVENT_WINDOW[0]))
                psth = normalize_psth_matrix(psth, target_duration)

                # match n_trials
                n_trials = min(psth.shape[1], df_trials.shape[0])
                psth = psth[:, :n_trials]
                df_trials = df_trials.iloc[:n_trials]

                aligned_all.append(psth)
                feedback_all.append(df_trials["feedbackType"].values)
                if "allContrasts" in df_trials.columns:
                    contrast_all.append(df_trials["allContrasts"].values)
                elif "signed_contrast" in df_trials.columns:
                    contrast_all.append(np.abs(df_trials["signed_contrast"].values))
                else:
                    contrast_all.append(np.zeros(n_trials))

        if not aligned_all:
            print(f"⚠️ No sessions for {NM}")
            continue

        psth_combined = np.concatenate(aligned_all, axis=1)
        feedback_combined = np.concatenate(feedback_all)
        contrast_combined = np.round(np.concatenate(contrast_all), 4)
        contrasts = np.sort(np.unique(contrast_combined))

        # baseline correction
        psth_combined = baseline_correct_per_trial(psth_combined, time_vector, BASELINE_WINDOW)

        mean_correct, sem_correct, mean_incorrect, sem_incorrect = {}, {}, {}, {}
        n_correct_total, n_incorrect_total = 0, 0

        for c in contrasts:
            idx_c = np.isclose(contrast_combined, c, atol=1e-4)
            idx_correct = (feedback_combined == 1) & idx_c
            idx_incorrect = (feedback_combined == -1) & idx_c

            psth_c_correct = psth_combined[:, idx_correct]
            psth_c_incorrect = psth_combined[:, idx_incorrect]

            mean_correct[c] = np.nanmean(psth_c_correct, axis=1)
            sem_correct[c] = sem(psth_c_correct, axis=1, nan_policy="omit")
            mean_incorrect[c] = np.nanmean(psth_c_incorrect, axis=1)
            sem_incorrect[c] = sem(psth_c_incorrect, axis=1, nan_policy="omit")

            n_correct_total += np.sum(idx_correct)
            n_incorrect_total += np.sum(idx_incorrect)

        # ===== Plot Correct / Incorrect =====
        for col_idx, (label, mean_dict, sem_dict, n_trials_total) in enumerate(zip(
                ["Future Correct", "Future Incorrect"],
                [mean_correct, mean_incorrect],
                [sem_correct, sem_incorrect],
                [n_correct_total, n_incorrect_total])):

            ax = axes[row_idx, col_idx]
            for j, c in enumerate(contrasts):
                color = colors_nm[j] if j < len(colors_nm) else colors_nm[-1]
                ax.plot(time_vector, mean_dict[c], lw=2.8, color=color, label=f"Contrast {c:.4f}".rstrip("0").rstrip("."))
                ax.fill_between(time_vector,
                                mean_dict[c] - sem_dict[c],
                                mean_dict[c] + sem_dict[c],
                                color=color, alpha=0.25)

            ax.axvline(0, color="black", lw=1.8, ls="--")
            ax.set_xlim(PERIEVENT_WINDOW)
            ax.set_ylim(YLIM)
            ax.set_title(label if row_idx == 0 else "", fontsize=13)
            ax.text(0.02, 0.95, f"n={n_trials_total}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=10, fontweight="bold", color="gray")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col_idx == 0:
                ax.set_ylabel(f"{NM}", fontsize=13, rotation=0, labelpad=20, va="center")

            if row_idx == 3:
                ax.set_xlabel("Time (s)", fontsize=12)
            else:
                ax.set_xticklabels([])

            if row_idx == 0 and col_idx == 1:
                ax.legend(frameon=False, fontsize=9, loc="upper right")

    # ----- Final touches -----
    fig.suptitle(f"PSTHs aligned to {EVENT.replace('_', ' ')} — baseline −0.05→0 s", fontsize=15, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    save_png = os.path.join(PSTH_DIR, f"AllNMs_grid_{EVENT}_baselinecorr.png")
    save_pdf = os.path.join(PSTH_DIR, f"AllNMs_grid_{EVENT}_baselinecorr.pdf")
    plt.savefig(save_png, dpi=300)
    plt.savefig(save_pdf, dpi=300)
    plt.close()
    print(f"✅ Saved summary grid for {EVENT} → {save_png}")
# %%


# %%
# %%
# =========================================================
# =========================================================
# =========================================================
# Psychometric curves for high-performance sessions only
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import norm
from scipy.optimize import minimize

# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
df_path = os.path.join(BASE_DIR, "df_goodSessions_highPerf.xlsx")
df_high = pd.read_excel(df_path)

MICE = {
    "DA":   ["ZFM-03447","ZFM-03448","ZFM-03450","ZFM-04026","ZFM-04019","ZFM-04022"],
    "5-HT": ["ZFM-03061","ZFM-03065","ZFM-03059","ZFM-03062","ZFM-04392","ZFM-05236","ZFM-05248","ZFM-05245","ZFM-05235"],
    "NE":   ["ZFM-04533","ZFM-04534","ZFM-06271","ZFM-06272","ZFM-06171","ZFM-06268","ZFM-06275"],
    "ACh":  ["ZFM-06305","ZFM-06946","ZFM-06948"],
}

PALETTE = {
    "DA":   "#D44836",
    "5-HT": "#7F4CC8",
    "NE":   "#2E84CC",
    "ACh":  "#45B437",
}

OUT_DIR = os.path.join(BASE_DIR, "psychometric_plots_highPerf")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def find_trials_file(base_dir, subject, date, region, eid):
    """Finds df_trials file ignoring numeric prefix."""
    for f in os.listdir(base_dir):
        if f.startswith("df_trials_") and subject in f and date in f and region in f and eid in f and f.endswith(".csv"):
            return os.path.join(base_dir, f)
    return None

def signed_contrast_x100(df):
    """Signed contrast ×100: (right - left)."""
    L = df["contrastLeft"].fillna(0).values
    R = df["contrastRight"].fillna(0).values
    return (R - L)

def stim_side(df):
    """+1 if right, -1 if left."""
    left_present  = df["contrastLeft"].notna().values
    right_present = df["contrastRight"].notna().values
    side = np.full(len(df), np.nan)
    side[(right_present) & (~left_present)] = +1
    side[(left_present) & (~right_present)] = -1
    return side

def session_prob_right_by_contrast(df):
    """Returns signed contrast (xc) and P(choice right) per contrast."""
    s = stim_side(df)
    fb = df["feedbackType"].values
    valid = (~np.isnan(s)) & np.isin(fb, [+1, -1])
    if not np.any(valid):
        return np.array([]), np.array([])

    sc = signed_contrast_x100(df)
    sc_v, s_v, fb_v = sc[valid], s[valid], fb[valid]
    chose_right = (fb_v * s_v) == +1

    xc = np.sort(np.unique(np.round(sc_v, 2)))
    pc = np.empty_like(xc, dtype=float)
    for i, c in enumerate(xc):
        m = np.isclose(sc_v, c, atol=1e-6)
        pc[i] = np.mean(chose_right[m]) if np.any(m) else np.nan
    return xc, pc

def interpolate_to_grid(xc, pc, xgrid):
    """Interpolate session data to common grid."""
    ok = ~np.isnan(pc)
    if np.sum(ok) < 2:
        return np.full_like(xgrid, np.nan)
    order = np.argsort(xc[ok])
    return np.interp(xgrid, xc[ok][order], pc[ok][order], left=np.nan, right=np.nan)

def smooth_curve(y, window=21, poly=3):
    """Apply Savitzky–Golay smoothing."""
    y2 = y.copy()
    ok = ~np.isnan(y2)
    if np.sum(ok) < window:
        return y2
    y2[ok] = savgol_filter(y2[ok], window_length=window, polyorder=poly)
    return y2

# ---- IBL-style psychometric fit ----
def psychometric_function(x, mu, sigma, gamma, lam):
    """Cumulative Gaussian psychometric function."""
    return gamma + (1 - gamma - lam) * norm.cdf(x, loc=mu, scale=sigma)

def fit_psychometric(x, y):
    """Maximum-likelihood fit of cumulative Gaussian to averaged curve."""
    x, y = np.asarray(x), np.asarray(y)
    ok = ~np.isnan(y)
    if np.sum(ok) < 4:
        return (np.nan, np.nan, np.nan, np.nan)
    x, y = x[ok], y[ok]
    y = np.clip(y, 1e-4, 1 - 1e-4)

    def nll(params):
        mu, sigma, gamma, lam = params
        if sigma <= 0 or gamma < 0 or lam < 0 or gamma + lam >= 1:
            return np.inf
        p = psychometric_function(x, mu, sigma, gamma, lam)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    init   = [0, 0.3, 0.05, 0.05]
    bounds = [(-1, 1), (1e-3, 5), (0, 0.4), (0, 0.4)]
    res = minimize(nll, init, bounds=bounds)
    return res.x if res.success else (np.nan, np.nan, np.nan, np.nan)

# =========================================================
# MAIN LOOP PER NM
# =========================================================
for NM, subjects in MICE.items():
    color = PALETTE[NM]
    xgrid = np.linspace(-1, 1, 201)
    session_curves, session_dots = [], []
    sess_used = 0

    # ---- iterate sessions ----
    for _, row in df_high.iterrows():
        subject, date, region, eid = row["subject"], str(row["date"])[:10], str(row["region"]), str(row["eid"])
        if subject not in subjects:
            continue
        fpath = find_trials_file(BASE_DIR, subject, date, region, eid)
        if not fpath:
            continue
        df = pd.read_csv(fpath)
        xc, pc = session_prob_right_by_contrast(df)
        if xc.size < 2:
            continue
        y = interpolate_to_grid(xc, pc, xgrid)
        y = smooth_curve(y, window=13, poly=2)
        session_curves.append(y)
        session_dots.append((xc, pc))
        sess_used += 1

    if sess_used == 0:
        print(f"⚠️ No sessions for {NM}")
        continue

    session_curves = np.array(session_curves)
    mean_curve = np.nanmean(session_curves, axis=0)
    mean_curve = smooth_curve(mean_curve, window=25, poly=3)

    # ---- IBL-style fit to mean curve ----
    mu, sigma, gamma, lam = fit_psychometric(xgrid, mean_curve)
    yfit = psychometric_function(xgrid, mu, sigma, gamma, lam)
    print(f"✅ {NM}: μ={mu:.3f}, σ={sigma:.3f}, γ={gamma:.3f}, λ={lam:.3f}")

    # =========================================================
    # PLOT
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 9), dpi=300)

    # faded per-session lines
    for y in session_curves:
        ax.plot(xgrid, y, color=color, alpha=0.45, lw=1.0)

    # scatter dots per session
    for xc, pc in session_dots:
        ax.scatter(xc, pc, color=color, alpha=0.45, s=30, edgecolor="none")

    # thick IBL-style fit line
    ax.plot(xgrid, yfit, color="black", lw=8, label="IBL-style psychometric fit")

    # cosmetics — untouched from your version
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1.0, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1.0])
    ax.set_xticklabels(["−1.0", "−0.25", " ", " ", "0", " ", " ", "0.25", "1.0"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Signed contrast (left | right)")
    ax.set_ylabel("Proportion of right choices")
    ax.set_title(f"Psychometric curve for {NM}  (N={sess_used} high-performance sessions)")
    ax.legend(frameon=False, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # optional: add small parameter box like IBL figures
    txt = f"μ = {mu:.3f}\nσ = {sigma:.3f}\nγ = {gamma:.3f}\nλ = {lam:.3f}"
    ax.text(0.75, 0.15, txt, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"psychometric_{NM}_IBLfit.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, f"psychometric_{NM}_IBLfit.pdf"), dpi=300)
    plt.show()

print("✅ Done — IBL-style psychometric fits saved in:", OUT_DIR)
# %%
""" PERFECT PSYCHOMETRICS """
# %%
# %%
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import norm
from scipy.optimize import minimize

# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = "/home/kceniabougrova/Downloads/good_sessions_outputs/"
df_path = os.path.join(BASE_DIR, "df_goodSessions_highPerf.xlsx")
df_high = pd.read_excel(df_path)

MICE = {
    "DA":   ["ZFM-03447","ZFM-03448","ZFM-03450","ZFM-04026","ZFM-04019","ZFM-04022"],
    "5-HT": ["ZFM-03061","ZFM-03065","ZFM-03059","ZFM-03062","ZFM-04392","ZFM-05236","ZFM-05248","ZFM-05245","ZFM-05235"],
    "NE":   ["ZFM-04533","ZFM-04534","ZFM-06271","ZFM-06272","ZFM-06171","ZFM-06268","ZFM-06275"],
    "ACh":  ["ZFM-06305","ZFM-06946","ZFM-06948"],
}

PALETTE = {
    "DA":   "#D44836",
    "5-HT": "#7F4CC8",
    "NE":   "#2E84CC",
    "ACh":  "#45B437",
}

OUT_DIR = os.path.join(BASE_DIR, "psychometric_plots_highPerf")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def find_trials_file(base_dir, subject, date, region, eid):
    """Finds df_trials file ignoring numeric prefix."""
    for f in os.listdir(base_dir):
        if f.startswith("df_trials_") and subject in f and date in f and region in f and eid in f and f.endswith(".csv"):
            return os.path.join(base_dir, f)
    return None

def signed_contrast_x100(df):
    """Signed contrast ×100: (right - left)."""
    L = df["contrastLeft"].fillna(0).values
    R = df["contrastRight"].fillna(0).values
    return (R - L)

def stim_side(df):
    """+1 if right, -1 if left."""
    left_present  = df["contrastLeft"].notna().values
    right_present = df["contrastRight"].notna().values
    side = np.full(len(df), np.nan)
    side[(right_present) & (~left_present)] = +1
    side[(left_present) & (~right_present)] = -1
    return side

def session_prob_right_by_contrast(df):
    """Returns signed contrast (xc) and P(choice right) per contrast."""
    s = stim_side(df)
    fb = df["feedbackType"].values
    valid = (~np.isnan(s)) & np.isin(fb, [+1, -1])
    if not np.any(valid):
        return np.array([]), np.array([])

    sc = signed_contrast_x100(df)
    sc_v, s_v, fb_v = sc[valid], s[valid], fb[valid]
    chose_right = (fb_v * s_v) == +1

    xc = np.sort(np.unique(np.round(sc_v, 2)))
    pc = np.empty_like(xc, dtype=float)
    for i, c in enumerate(xc):
        m = np.isclose(sc_v, c, atol=1e-6)
        pc[i] = np.mean(chose_right[m]) if np.any(m) else np.nan
    return xc, pc

def interpolate_to_grid(xc, pc, xgrid):
    """Interpolate session data to common grid."""
    ok = ~np.isnan(pc)
    if np.sum(ok) < 2:
        return np.full_like(xgrid, np.nan)
    order = np.argsort(xc[ok])
    return np.interp(xgrid, xc[ok][order], pc[ok][order], left=np.nan, right=np.nan)

def smooth_curve(y, window=21, poly=3):
    """Apply Savitzky–Golay smoothing."""
    y2 = y.copy()
    ok = ~np.isnan(y2)
    if np.sum(ok) < window:
        return y2
    y2[ok] = savgol_filter(y2[ok], window_length=window, polyorder=poly)
    return y2

# ---- IBL-style psychometric fit ----
def psychometric_function(x, mu, sigma, gamma, lam):
    """Cumulative Gaussian psychometric function."""
    return gamma + (1 - gamma - lam) * norm.cdf(x, loc=mu, scale=sigma)

def fit_psychometric(x, y):
    """Maximum-likelihood fit of cumulative Gaussian to averaged curve."""
    x, y = np.asarray(x), np.asarray(y)
    ok = ~np.isnan(y)
    if np.sum(ok) < 4:
        return (np.nan, np.nan, np.nan, np.nan)
    x, y = x[ok], y[ok]
    y = np.clip(y, 1e-4, 1 - 1e-4)

    def nll(params):
        mu, sigma, gamma, lam = params
        if sigma <= 0 or gamma < 0 or lam < 0 or gamma + lam >= 1:
            return np.inf
        p = psychometric_function(x, mu, sigma, gamma, lam)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    init   = [0, 0.3, 0.05, 0.05]
    bounds = [(-1, 1), (1e-3, 5), (0, 0.4), (0, 0.4)]
    res = minimize(nll, init, bounds=bounds)
    return res.x if res.success else (np.nan, np.nan, np.nan, np.nan)

# =========================================================
# MAIN LOOP PER NM
# =========================================================
for NM, subjects in MICE.items():
    color = PALETTE[NM]
    xgrid = np.linspace(-1, 1, 201)
    session_curves, session_dots = [], []
    sess_used = 0

    # ---- iterate sessions ----
    for _, row in df_high.iterrows():
        subject, date, region, eid = row["subject"], str(row["date"])[:10], str(row["region"]), str(row["eid"])
        if subject not in subjects:
            continue
        fpath = find_trials_file(BASE_DIR, subject, date, region, eid)
        if not fpath:
            continue
        df = pd.read_csv(fpath)
        xc, pc = session_prob_right_by_contrast(df)
        if xc.size < 2:
            continue
        y = interpolate_to_grid(xc, pc, xgrid)
        y = smooth_curve(y, window=13, poly=2)
        session_curves.append(y)
        session_dots.append((xc, pc))
        sess_used += 1

    if sess_used == 0:
        print(f"⚠️ No sessions for {NM}")
        continue

    session_curves = np.array(session_curves)
    mean_curve = np.nanmean(session_curves, axis=0)
    mean_curve = smooth_curve(mean_curve, window=25, poly=3)

    # ---- IBL-style fit to mean curve ----
    mu, sigma, gamma, lam = fit_psychometric(xgrid, mean_curve)
    yfit = psychometric_function(xgrid, mu, sigma, gamma, lam)
    print(f"✅ {NM}: μ={mu:.3f}, σ={sigma:.3f}, γ={gamma:.3f}, λ={lam:.3f}")

    # =========================================================
    # PLOT
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 9), dpi=300)

    # faded per-session lines
    for y in session_curves:
        ax.plot(xgrid, y, color=color, alpha=0.75, lw=1.0)

    # scatter dots per session
    for xc, pc in session_dots:
        ax.scatter(xc, pc, color=color, alpha=0.75, s=35, edgecolor="none")

    # thick IBL-style fit line
    ax.plot(xgrid, yfit, color="black", lw=8, label="Average psychometric fit")

    # cosmetics — untouched from your version
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1.0, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1.0])
    ax.set_xticklabels(["−1.0", "−0.25", " ", " ", "0", " ", " ", "0.25", "1.0"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Signed contrast (left | right)")
    ax.set_ylabel("Proportion of right choices")
    ax.set_title(f"Psychometric curve for {NM}  (N={sess_used} high-performance sessions)")
    ax.legend(frameon=False, loc="lower right")
    # --- add a "single session" line proxy for legend ---
    from matplotlib.lines import Line2D
    session_line = Line2D([0], [0], color=color, lw=1.5, alpha=0.5, label="Single session")
    mean_line    = Line2D([0], [0], color="black", lw=8, label="IBL-style psychometric fit")
    ax.legend(handles=[session_line, mean_line], frameon=False, loc="lower right")


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # optional: add small parameter box like IBL figures
    txt = f"μ = {mu:.3f}\nσ = {sigma:.3f}\nγ = {gamma:.3f}\nλ = {lam:.3f}"
    ax.text(0.75, 0.25, txt, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"psychometric_{NM}_IBLfit.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, f"psychometric_{NM}_IBLfit.pdf"), dpi=300)
    plt.show()

print("✅ Done — IBL-style psychometric fits saved in:", OUT_DIR)
# %%
