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
    "quiescenceTime","quiescencePeriod","stimSide","allContrasts",
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
df_results.to_csv(os.path.join(BASE_DIR, f"ridge_results_allSessions_alpha{alpha}.csv"), index=False)

# Plot distribution of R² per signal
plt.figure(figsize=(7,4))
for t in targets:
    plt.hist(df_results[df_results["target"]==t]["r2"], bins=20, alpha=0.5, label=t)
plt.xlabel("R²")
plt.ylabel("Sessions")
plt.title("Explained variance of neural signals by behavioral model (α=5)")
plt.legend()
plt.tight_layout()
plt.show()

# %%





# %%
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