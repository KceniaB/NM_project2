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