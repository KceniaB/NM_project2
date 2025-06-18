"""
KB 20250618
from NM_PROJECT 
move files to corresponding date folders - works! 
""" 

# %%
"""
Step 1. Check which files have photometry folders for the: 
    - raw_photometry
    - region photometry
"""
import os
import pandas as pd

base_path = "/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/data/one/mainenlab/Subjects"

results = []
multiple_sessions = []

for subject in os.listdir(base_path):
    subject_path = os.path.join(base_path, subject)
    if not os.path.isdir(subject_path):
        continue

    for date in os.listdir(subject_path):
        date_path = os.path.join(subject_path, date)
        if not os.path.isdir(date_path):
            continue

        session_dirs = [s for s in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, s))]
        if len(session_dirs) != 1:
            multiple_sessions.append({
                "path": date_path,
                "subject": subject,
                "date": date
            })
            continue

        session = session_dirs[0]
        session_path = os.path.join(date_path, session)

        # Check folder1
        folder1_path = os.path.join(session_path, "raw_photometry_data", "raw_photometry.csv")
        folder1_exists = os.path.isfile(folder1_path)

        # Check folder2 - alf/RegionXG/
        alf_path = os.path.join(session_path, "alf")
        found_region = False
        if os.path.isdir(alf_path):
            for region in os.listdir(alf_path):
                if region.startswith("Region") and region.endswith("G"):
                    found_region = True
                    region_path = os.path.join(alf_path, region, "raw_photometry.pqt")
                    folder2_exists = os.path.isfile(region_path)
                    results.append({
                        "path": session_path,
                        "subject": subject,
                        "date": date,
                        "region": region,
                        "folder1": folder1_exists,
                        "folder2": folder2_exists
                    })

        if not found_region:
            results.append({
                "path": session_path,
                "subject": subject,
                "date": date,
                "region": None,
                "folder1": folder1_exists,
                "folder2": False
            })

# Save results
df_main = pd.DataFrame(results)
df_multiple_sessions = pd.DataFrame(multiple_sessions)

df_main.to_csv("folder_check_results.csv", index=False)
df_multiple_sessions.to_csv("multiple_sessions_detected.csv", index=False)

print("Finished. Results saved to 'folder_check_results.csv' and 'multiple_sessions_detected.csv'")


# %%
"""
Step 2. Keep the important mice 
    - remove: 'ZFM-02369', 'ZFM-05239', 'ZFM-05244', 
       'ZFM-05247', 'ZFM-08652', 'ZFM-08751', 'ZFM-08757', 
       'ZFM-08827', 'ZFM-08828', 'ZM_3003' 

Note: 
    - not all the behavior sessions might be here
        => so lets check what are the photometry sessions that we have 
""" 

import os
import pandas as pd
from datetime import datetime

# Load df_main (from the previous step)
# df_main = pd.read_csv("folder_check_results.csv")

# Define folders A and B
folder_A = "/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/data/external_drive"
folder_B = "/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/data/external_drive/HCW_S2_23082021/Sorted_Files"

def list_valid_dates(base_folder):
    valid_dates = set()
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            try:
                datetime.strptime(folder, "%Y-%m-%d")
                valid_dates.add(folder)
            except ValueError:
                continue
    return valid_dates

dates_A = list_valid_dates(folder_A)
dates_B = list_valid_dates(folder_B)

main_dates = set(df_main['date'].unique())
external_dates = dates_A.union(dates_B)

# Map where each date is present
date_source_map = {}
for date in main_dates:
    in_A = date in dates_A
    in_B = date in dates_B
    if in_A and in_B:
        date_source_map[date] = "A+B"
    elif in_A:
        date_source_map[date] = "A"
    elif in_B:
        date_source_map[date] = "B"
    else:
        date_source_map[date] = "no"

df_main["present"] = df_main["date"].map(date_source_map)

# Find external-only dates
missing_dates = external_dates - main_dates
missing_df = pd.DataFrame(sorted(missing_dates), columns=["date"])
missing_df["A"] = missing_df["date"].apply(lambda d: "here" if d in dates_A else "")
missing_df["B"] = missing_df["date"].apply(lambda d: "here" if d in dates_B else "")

# Save outputs
# df_main.to_csv("updated_df_main.csv", index=False)
# missing_df.to_csv("external_only_dates.csv", index=False)

# print("✔ Done! Files saved: 'updated_df_main.csv' and 'external_only_dates.csv'")




# %%
"""
3. Summary so far: 
    - df_main - if a certain behavior subject and date have or not photometry raw data within the behav folders 
        path    subject     date    region	    folder1	    folder2	    present
        => the ones that dont have are the ones to work with from here on 

    - df_multiple_sessions - sessions with more than 1 behavior folder (example 001, 002, 003, ...) 
        path	subject	    date
        => to debug - maybe remove the sessions where there is not much behav, or concatenate sessions that were cut due to bugs 

    - missing_df - dates that are in the photometry folders but don't have a corresponding behavior folder for any subject 
        {'2023-01-16',
        '2023-04-11',
        '2023-05-12',
        '2023-05-31',
        '2023-06-05',
        '2023-06-06',
        '2023-06-13',
        '2023-11-21',
        '2024-01-25'} 
    
    
    - df_main["present"] 
        present
        A     1249 - those are in A
        B      533 - those are in B
        no     483 - most likely no photometry 
"""

"""
4. Lets sort by subject and date
"""
df_main.sort_values(by=["subject", "date"], inplace=True)

# df_main.to_csv("updated_df_main.csv", index=False)





# %%

import os
import shutil
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from functions_nm import load_trials 
# import kcenia as kcenia #added 16092024, works, it seems? 
# import iblphotometry.kcenia as kcenia
import ibldsp.utils
from pathlib import Path
# import iblphotometry.plots
# import iblphotometry.dsp
from brainbox.io.one import SessionLoader 
import scipy.signal
import ibllib.plots
from one.api import ONE #always after the imports 
# one = ONE(cache_dir="/mnt/h0/kb/data/one")
one = ONE()
ROOT_DIR = Path('/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/') 

destination_base_path = os.path.join(ROOT_DIR, 'kb/data/external_drive/HCW_S2_23082021/Sorted_Files') 

#%%
"""
#################### path to store the photometry file ####################
""" 
# df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1 = read this: '/home/kceniabougrova/Downloads/df_new_restructured.csv' 
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str) 

def get_eid(rec): 
    eids = one.search(subject=rec.subject, date=rec.date) 
    eid = eids[0]
    return eid

def get_regions(rec): 
    regions = rec.region
    return regions 

def get_nph(source_path, rec): 
    # source_folder = (f"/home/kceniabougrova/Documents/nph/{rec.date}/") 
    source_folder = source_path
    df_nph = pd.read_csv(source_folder+f"raw_photometry{rec.nph_file}.csv") 
    return df_nph  

# %%
import os
import pandas as pd

base_path = "/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/data/one/mainenlab/Subjects"

results = []
multiple_sessions = []

for subject in os.listdir(base_path):
    subject_path = os.path.join(base_path, subject)
    if not os.path.isdir(subject_path):
        continue

    for date in os.listdir(subject_path):
        date_path = os.path.join(subject_path, date)
        if not os.path.isdir(date_path):
            continue

        session_dirs = [s for s in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, s))]
        if len(session_dirs) != 1:
            multiple_sessions.append({
                "path": date_path,
                "subject": subject,
                "date": date
            })
            continue

        session = session_dirs[0]
        session_path = os.path.join(date_path, session)

        # Check folder1
        folder1_path = os.path.join(session_path, "raw_photometry_data", "raw_photometry.csv")
        folder1_exists = os.path.isfile(folder1_path)

        # Check folder2 - alf/RegionXG/
        alf_path = os.path.join(session_path, "alf")
        found_region = False
        if os.path.isdir(alf_path):
            for region in os.listdir(alf_path):
                if region.startswith("Region") and region.endswith("G"):
                    found_region = True
                    region_path = os.path.join(alf_path, region, "raw_photometry.pqt")
                    folder2_exists = os.path.isfile(region_path)
                    results.append({
                        "path": session_path,
                        "subject": subject,
                        "date": date,
                        "region": region,
                        "folder1": folder1_exists,
                        "folder2": folder2_exists
                    })

        if not found_region:
            results.append({
                "path": session_path,
                "subject": subject,
                "date": date,
                "region": None,
                "folder1": folder1_exists,
                "folder2": False
            })

# Save results
df_main = pd.DataFrame(results)
df_multiple_sessions = pd.DataFrame(multiple_sessions)

df_main.to_csv("folder_check_results.csv", index=False)
df_multiple_sessions.to_csv("multiple_sessions_detected.csv", index=False)

print("Finished. Results saved to 'folder_check_results.csv' and 'multiple_sessions_detected.csv'")
