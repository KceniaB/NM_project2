""" 
KB 
24-August-2025
picking all the sessions under the project 'ibl_fibrephotometry' 
"""
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


table_path = '/home/kceniabougrova/Downloads/KB_sessions_insertions_map - upload (1).csv'
sessions_list = pd.read_csv(table_path)


eids = one.search(project='ibl_fibrephotometry')
print("EIDs found:", eids)

# To fetch additional metadata about each session
eids, info = one.search(project='ibl_fibrephotometry', details=True)

# Convert eids to plain strings
eid_list = [str(eid) for eid in eids]

# Build DataFrame
df = pd.DataFrame(info)
df.insert(0, "eid", eid_list)

print(df.head())

# for eid, session_info in zip(eids, info):
#     print(f"Experiment ID: {eid}, Lab: {session_info.get('lab')}, Subject: {session_info.get('subject')}")

df



df.to_csv("ibl_fibrephotometry_sessions.csv", index=False)


#%%
"""
filtering by sessions ran by Kcenia / Kcenia's mice - done in the google sheet 
""" 
table_path2 = '/home/kceniabougrova/Downloads/Untitled spreadsheet - ibl_fibrephotometry_sessions.csv'
sessions_list2 = pd.read_csv(table_path2)

# %%
sessions_list2["present_in_list"] = sessions_list2["eid"].isin(sessions_list["eid"]).astype(int)
sessions_list2

# Count how many 0s and 1s
counts = sessions_list2["flag"].value_counts()
print(counts)

"""
present_in_list
0    1488
1     837
"""


#%%
# LONG RUN (86minutes) 

def update_sessions_list(sessions_list2):
    # Add new column for n_trials if it doesn't exist
    if "n_trials" not in sessions_list2.columns:
        sessions_list2["n_trials"] = None

    for idx, row in sessions_list2.iterrows():
        if row["present_in_list"] == 1:  # only loop over 1s
            eid = row["eid"]

            try:
                # Load data
                df_trials, subject, session_date = load_trials_updated(eid)

                # Count trials
                n_trials = len(df_trials)

                # Save back into dataframe
                sessions_list2.at[idx, "n_trials"] = n_trials

                # Check subject match
                if row["subject"] != subject:
                    print(f"⚠️ Subject mismatch for eid {eid}: df={subject}, list={row['subject']}")

                # Check date match (convert both to string if needed)
                if str(row["date"]) != str(session_date):
                    print(f"⚠️ Date mismatch for eid {eid}: df={session_date}, list={row['date']}")

            except Exception as e:
                print(f"❌ Error processing eid {eid}: {e}")

    return sessions_list2


# Usage
sessions_list2 = update_sessions_list(sessions_list2)

#%%
sessions_list2.to_csv("ibl_fibrephotometry_sessions_checked.csv", index=False)

""" 
len(sessions_list2.n_trials.unique()) 
545
"""
# %% 
""" LONGER RUN (168 minutes)"""

# Prepare storage for results
trial_counts = []
missing_trials = []

# Loop through only rows with present_in_list == 0
for idx, row in sessions_list2.iterrows():
    if row["present_in_list"] == 0:
        eid = row["eid"]
        subject = row.get("subject", None)
        date = row.get("date", None)

        try:
            trials = load_trials(eid)  # your function
            if trials is not None and len(trials) > 0:
                trial_counts.append(len(trials))
            else:
                trial_counts.append(None)
                missing_trials.append([eid, subject, date])
        except Exception as e:
            trial_counts.append(None)
            missing_trials.append([eid, subject, date])
    else:
        trial_counts.append(None)  # skip rows with present_in_list == 1

# Add as new column
sessions_list2["trial_count"] = trial_counts

print("✅ Finished checking trials")
print(f"Total missing trial datasets: {len(missing_trials)}")

sessions_list2.to_csv("ibl_fibrephotometry_sessions_checked_longer.csv", index=False)





#%%

import pandas as pd

def merge_trials(row):
    t1, t2 = row["n_trials"], row["n_trials2"]
    if pd.isna(t1) and pd.isna(t2):
        return None
    if pd.isna(t1):
        return t2
    if pd.isna(t2):
        return t1
    if t1 == t2:
        return t1
    # Raise error if mismatch
    raise ValueError(f"Mismatch in trials for eid {row.name}: n_trials={t1}, n_trials2={t2}")

# Drop unwanted column
sessions_list2 = sessions_list2.drop(columns=["trial_count"])

# Create merged column
sessions_list2["n_trialstotal"] = sessions_list2.apply(merge_trials, axis=1)

# len(sessions_list2.n_trialstotal.unique())
# 856 




#%% 
df_missing_trials = sessions_list2[sessions_list2["n_trialstotal"].isna()].copy()

# 237 missing, no n_trials info 

#%% 
""" 
if we want to find those n_trials, we must use one.load_dataset(eid, '_iblrig_taskData.raw.jsonable')
"""

# Make a copy just in case
sessions_fixed = sessions_list2.copy()

failed = []

for idx, row in sessions_fixed[sessions_fixed["n_trialstotal"].isna()].iterrows():
    eid = row["eid"]  # or whatever your eid column is called
    
    try:
        task_data = one.load_dataset(eid, "_iblrig_taskData.raw.jsonable")
        sessions_fixed.at[idx, "n_trialstotal"] = len(task_data)
    except Exception as e:
        print(f"⚠️ Could not load taskData for {eid}: {e}")
        failed.append((eid, row["subject"], row["date"]))

print(f"Done. Filled {sessions_fixed['n_trialstotal'].notna().sum()} rows. "
      f"{len(failed)} rows failed.")

#%%
""" _iblrig_NPH_tasks_trainingChoiceWorld26.5.2 """
# ⚠️ Could not load taskData for 8b17adb2-289a-4992-99c5-c3cf2adb8059: Dataset "_iblrig_taskData.raw.jsonable" not found 

""" _iblrig_NPH_tasks_trainingChoiceWorld2 """
# ⚠️ Could not load taskData for fc04c4b8-4f4d-49ef-a311-19b0771349d5: Dataset "_iblrig_taskData.raw.jsonable" not found 

""" HISTOLOGY """
# ⚠️ Could not load taskData for ee8ecff7-658b-4d14-a025-fbf3e64253c2: Dataset "_iblrig_taskData.raw.jsonable" not found 

""" _iblrig_tasks_biasedChoiceWorld_RPE6.5.3 """
# ⚠️ Could not load taskData for 07841400-20a7-47a7-ab0b-4cc1be4c1e48: Dataset "_iblrig_taskData.raw.jsonable" not found 

""" _iblrig_tasks_biasedChoiceWorld_RPE6.5.3 """
# ⚠️ Could not load taskData for f5414590-e877-41eb-9f64-362357535724: Dataset "_iblrig_taskData.raw.jsonable" not found 

""" _iblrig_tasks_biasedChoiceWorld_RPE6.5.3 """
# ⚠️ Could not load taskData for 92005744-7b01-406f-9913-8d79da11736c: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 

""" _iblrig_tasks_biasedChoiceWorld_NPH6.5.3 """
# ⚠️ Could not load taskData for 90678f5b-d509-4d7b-9374-981b314675b2: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 

""" _iblrig_tasks_biasedChoiceWorld_NPH6.5.3 """
# ⚠️ Could not load taskData for 6a9ec3a7-fa68-473a-ab0e-43a3fe5de0a1: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 

""" _iblrig_tasks_biasedChoiceWorld_NPH6.5.3 """
# ⚠️ Could not load taskData for cf47affa-7004-4da3-9518-09b2bb9e952c: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 

""" _iblrig_tasks_biasedChoiceWorld_NPH6.5.3 """
# ⚠️ Could not load taskData for 7eb5ed06-a9b7-4008-829e-252549102bbe: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 

""" HISTOLOGY """
# ⚠️ Could not load taskData for 8830bc75-617f-4854-b51d-10cd4231ac8b: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
# ⚠️ Could not load taskData for aa1daa19-3fd6-4f52-aa79-c5084f4f7a56: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 

""" _iblrig_tasks_biasedChoiceWorld6.6.1 """
# ⚠️ Could not load taskData for b625df85-cd17-45b3-b921-262134237331: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 

""" _iblrig_tasks_biasedChoiceWorld6.6.1 """
# ⚠️ Could not load taskData for ae827174-fb6b-41d1-933b-a71d8b881cc8: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 

""" _iblrig_tasks_trainingChoiceWorld """
# ⚠️ Could not load taskData for 2a6e2845-599f-4ba7-bb2c-d3f4b67a119c: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 

""" HISTOLOGY """
# ⚠️ Could not load taskData for ccb1cf64-718f-48c5-836d-b261a52df85f: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
# ⚠️ Could not load taskData for 5150a4a6-e845-4332-bfc2-daca873f5d1e: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
# ⚠️ Could not load taskData for b18a24cc-e970-4ddb-8e57-162c4166fb46: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
# ⚠️ Could not load taskData for 41bde3e6-ff05-428d-b4a7-38834ecade7a

""" _iblrig_NPH_tasks_trainingChoiceWorld2 """
# ⚠️ Could not load taskData for 8b52ce8a-40b4-4c16-87ff-df7c9d7578c4: object of type 'NoneType' has no len()

""" _iblrig_tasks_biasedChoiceWorld """
# ⚠️ Could not load taskData for eb41a8af-4f5c-40c6-926b-8def5941e0e8: Dataset "_iblrig_taskData.raw.jsonable" not found 

""" PASSIVE """
# ⚠️ Could not load taskData for 746d1803-9230-4c38-b17f-799fea5cbb3d: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 

""" PASSIVE """
# ⚠️ Could not load taskData for 5901a440-948c-478c-b270-5910a705f2d6: Dataset "_iblrig_taskData.raw.jsonable" not found 

""" HISTOLOGY """
# ⚠️ Could not load taskData for ae420e9b-3cfb-4125-8a5d-f978c1221395: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
# ⚠️ Could not load taskData for fff5267a-a21a-4e55-8099-1376227bdccd: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  Th
# ⚠️ Could not load taskData for 96d92cb2-e6c7-4f83-9b75-f221bac0a934: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The A
# ⚠️ Could not load taskData for eb12a797-4ec3-43cd-a92b-8b1dcf0d8d58: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
# ⚠️ Could not load taskData for dcc2e8c9-0ff6-4707-ab4a-e1b1073d1c98: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF obje

""" BIASED """
# ⚠️ Could not load taskData for fabee777-2075-4595-b510-e60f4e376113: Dataset "_iblrig_taskData.raw.jsonable" not found 

""" HISTOLOGY """
# ⚠️ Could not load taskData for 1793daf3-49a7-4bd8-ae77-565da354d8cf: Dataset "_iblrig_taskData.raw.jsonable" not found 



""" HISTOLOGY """
# ⚠️ Could not load taskData for 5250e513-5eb7-475e-9bd4-0c6d37390bbb: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
# ⚠️ Could not load taskData for 25e10daa-9dd8-4f70-9095-0ebc0eba4794: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
# (S3) /home/kceniabougrova/Downloads/ONE/alyx.internationalbrainlab.org/mainenlab/Subjects/ZFM-05245/2023-08-25/001/raw_task_data_00/_iblrig_taskData.raw.jsonable: 100%|██████████| 18.9M/18.9M [00:03<00:00, 5.77MB/s]

""" _iblrig_tasks_trainingChoiceWorld8 """ 
# ⚠️ Could not load taskData for 16a9a0e6-d627-4c62-b031-b23f175d6e26: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 

"""HISTOLOGY """ 
# ⚠️ Could not load taskData for a019efe8-34c6-4c5a-a58b-b6386468ec64: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
# (S3) /home/kceniabougrova/Downloads/ONE/alyx.internationalbrainlab.org/mainenlab/Subjects/ZFM-05235/2023-09-07/001/raw_task_data_00/_iblrig_taskData.raw.jsonable: 100%|██████████| 10.9M/10.9M [00:03<00:00, 3.02MB/s]
# ⚠️ Could not load taskData for 1f2b6bb1-47e0-4b5b-a1a9-348d1f30aeae: Dataset "_iblrig_taskData.raw.jsonable" not found 
#  The 


⚠️ Could not load taskData for a9658904-9c70-437d-b3e1-d095d5111534: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for c4a4dd61-aa7a-4ffb-9897-ab69efe030d1: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for dd4510e5-27ab-4dd5-87e3-de8adafabc7c: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
(S3) /home/kceniabougrova/Downloads/ONE/alyx.internationalbrainlab.org/mainenlab/Subjects/ZFM-05235/2023-11-02/001/raw_task_data_00/_iblrig_taskData.raw.jsonable: 100%|██████████| 16.1M/16.1M [00:02<00:00, 6.00MB/s]
⚠️ Could not load taskData for a83af345-4e8a-4008-8182-9609344f6ff5: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for ae932e24-56bd-4d40-8e8d-9ce6c1a78b4d: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for 8379f501-82e0-46be-a17f-a50ced2365dd: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The A


⚠️ Could not load taskData for 70d36f47-3619-4a29-a54e-f73c40f9cba4: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for 8f0c6cc0-d295-43ee-a1ee-9c8cb3761048: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for d8831e74-82bc-421a-85c9-9801e0d3c514: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for 1868827e-c675-4a7f-84b0-04e52d5d7a64: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for d7748672-8ee1-4559-958e-879020827924: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The A




⚠️ Could not load taskData for c60ff538-f8d9-4f67-a901-d0e965eabcab: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for 3eb27096-389a-47ad-97d9-0f0c91b101ea: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for a5cb21ee-edc3-4779-93a3-ffaa9e59db2e: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for e9e27fb1-fc7d-4449-b231-7c420f2d5422: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for 89bf323f-4d0b-4204-929f-e89c724de960: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for aa00b68c-66d3-4edb-931e-04e17222ed7d: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for 6b064c4c-66ee-4f56-a9a8-04115b363062: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for 3d70b129-804d-4612-8cbd-30b7c03a326f: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for 68d7f913-f4fa-4f2d-97fc-34539a5ac2be: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for c49fc113-999c-49d4-b5b9-8ab825a4fe82: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for afc11ae0-6e3e-489e-ab4d-9ef6b37a86d7: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for 553542d6-3ff4-4a8c-89fe-32389a5b6df5: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for 97f2e0bf-8bcb-481e-9f68-a327ce628421: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for d1329358-6738-44c7-9e6c-7560bb6765f4: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for ed37eca4-3e0b-4d4e-8ac7-2759990a9ab7: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for dd96055c-7337-4b83-a3bd-0d831c836a75: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
⚠️ Could not load taskData for baa0857c-b891-46ca-912c-34a56afc18ca: Dataset "_iblrig_taskData.raw.jsonable" not found 
 The ALF object was not found.  This may occur if the object or namespace or incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be found with the filters `object="trials", namespace="ibl"` 
Done. Filled 2262 rows. 63 rows failed.




#%%
""" 
sessions_fixed is the new dataframe with n_trialstotal column filled in where possible
""" 
sessions_fixed["n_trialstotal"].isna().sum()
# 63 many of which are HISTOLOGY
sessions_fixed.to_csv("ibl_fibrephotometry_sessions_checked_longer_filled.csv", index=False)


# %%
cols_to_show = ["task_protocol", "subject", "date",
                "present_in_list", "n_trials", "n_trials2", "n_trialstotal"]

mask = (
    (sessions_fixed["present_in_list"] == 1) &
    (sessions_fixed["n_trials"].isna()) &
    (sessions_fixed["n_trials2"].isna()) &
    (sessions_fixed["n_trialstotal"].notna())
)

print(sessions_fixed.loc[mask, cols_to_show])

#                                       task_protocol    subject        date  \
# 33    _iblrig_NPH_tasks_habituationChoiceWorld6.4.2  ZFM-03061  2021-08-27   
# 37    _iblrig_NPH_tasks_habituationChoiceWorld6.4.2  ZFM-03061  2021-08-30   
# 41    _iblrig_NPH_tasks_habituationChoiceWorld6.4.2  ZFM-03061  2021-08-31   
# 273          _iblrig_NPH_tasks_trainingChoiceWorld2  ZFM-03448  2021-11-21   
# 278          _iblrig_NPH_tasks_trainingChoiceWorld2  ZFM-03450  2021-11-24   
# 1106  _iblrig_NPH_tasks_habituationChoiceWorld7.0.4  ZFM-05236  2022-11-05   
# 1272  _iblrig_tasks_biasedChoiceWorld_ephyssessions  ZFM-04022  2023-01-11   
# 1304           _iblrig_tasks_habituationChoiceWorld  ZFM-03450  2023-01-18   

#       present_in_list n_trials n_trials2  n_trialstotal  
# 33                  1     None      None           91.0  
# 37                  1     None      None          141.0  
# 41                  1     None      None          222.0  
# 273                 1     None      None          470.0  
# 278                 1     None      None           88.0  
# 1106                1     None      None          126.0  
# 1272                1     None      None          337.0  
# 1304                1     None      None          272.0  