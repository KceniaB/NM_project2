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


""" 
5. Add data - 1st manually 
    - lets pick the None region rows and the A or B present ones
"""
filtered_df = df_main[df_main['region'].isna() & df_main['present'].isin(["A", "B"])]

"""
# this gives 985 rows!!! 
# but a day can have a session for one subject but not for another 
# so what data is needed in order to have the photometry in the behavior files? 
    1. have subject, date, region, photometry file and BNC file 
    2. fill this data 
    3. loop the code below, which must: 
        i. read the df with the info 
        ii. have the functions ready 
        iii. tbpod must be the behavior event to align the data 
        iv. iup is data from the BNC input number 
        v. 
"""

# %%
""" 
5.2 Let's try with a specific example 
"""

import sys
import os

# ZFM-03061	2021-09-21	4	50fd9088-2a75-4821-803b-cccbcf6e8b08	probe04	yes	yes		1
# ZFM-03061	2021-09-21	6	50fd9088-2a75-4821-803b-cccbcf6e8b08	probe05	yes	yes		1
path_nph = '/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/data/external_drive/HCW_S2_23082021/Sorted_Files/2021-09-21/PhotometryData_M1_M4_TCW_S18_21Sep2021.csv'
path_bnc = '/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/data/external_drive/HCW_S2_23082021/Sorted_Files/2021-09-21/DI1_M1_M4_TCW_S18_21Sep2021.csv'
subject = 'ZFM-03061'
date = '2021-09-21'
regions = 'Region4G'
regions2 = 'Region6G'
eid = '50fd9088-2a75-4821-803b-cccbcf6e8b08'
df_nph = pd.read_csv(path_nph) 


sl = SessionLoader(one=one, eid=eid) 
sl.load_trials()
df_trials = sl.trials #trials table
tbpod = sl.trials['stimOnTrigger_times'].values #bpod TTL times



# %%
"""
#################### IMPORTS ####################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ibldsp.utils
from pathlib import Path
from brainbox.io.one import SessionLoader 
import scipy.signal
import ibllib.plots
from one.api import ONE #always after the imports 
one = ONE()


#%%
"""
#################### path to store the photometry file ####################
""" 
dtype = {'nph_file': int, 'nph_bnc': int, 'region': int}
# df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1 = pd.read_excel('/mnt/h0/kb/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str) 

def get_eid(rec): 
    eids = one.search(subject=rec.mouse, date=rec.date) 
    eid = eids[0]
    return eid

def get_regions(rec): 
    regions = [f"Region{rec.region}G"] 
    return regions 

def get_nph(source_path, rec): 
    # source_folder = (f"/home/kceniabougrova/Documents/nph/{rec.date}/") 
    source_folder = source_path
    df_nph = pd.read_csv(source_folder+f"raw_photometry{rec.nph_file}.csv") 
    return df_nph  








# %%





# %% 
# %%
# %%
# %%
def get_ttl(df_DI0): 
    if 'Value.Value' in df_DI0.columns: #for the new ones
        df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
    elif 'Timestamp' in df_DI0.columns: 
        df_DI0["Timestamp"] = df_DI0["Timestamp"] #for the old ones #KB added 20082024
    else:
        df_DI0["Timestamp"] = df_DI0["Seconds"] #for the old ones
    #use Timestamp from this part on, for any of the files
    raw_phdata_DI0_true = df_DI0[df_DI0.Value==True]
    df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
    # raw_phdata_DI0_true = pd.DataFrame(df_DI0.Timestamp[df_DI0.Value==True], columns=['Timestamp'])
    df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True) 
    tph = df_raw_phdata_DI0_T_timestamp.values[:, 0] 
    return tph 
df_nphttl = pd.read_csv(path_bnc) 

tph = get_ttl(df_nphttl)    
iup = tph
# tph = (df_ph['Timestamp'].values[iup] + df_ph['Timestamp'].values[iup - 1]) / 2 #nph TTL times computed for the midvalue 
tph = iup
fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True) #interpolation 
if len(tph)/len(tbpod) < .9: 
    print("mismatch in sync, will try to add ITI duration to the sync")
    tbpod = np.sort(np.r_[
        df_trials['intervals_0'].values,
        df_trials['intervals_1'].values - 1,  # here is the trick
        df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
    )
    fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)
    if len(tph)/len(tbpod) > .9:
        print("still mismatch, maybe this is an old session")
        tbpod = np.sort(np.r_[df_trials['stimOnTrigger_times'].values])
        fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True, return_indices=True) 
        assert len(iph)/len(tbpod) > .9
        print("recovered from sync mismatch, continuing #2")
assert abs(drift_ppm) < 100, "drift is more than 100 ppm"
    
df_ph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_ph["Timestamp"]) 

fcn_nph_to_bpod_times(df_ph["Timestamp"])

df_ph["Timestamp"]


# Assuming tph contains the timestamps in seconds
tbpod_start = tbpod[0] - 30  # Start time, 100 seconds before the first tph value
tbpod_end = tbpod[-1] + 30   # End time, 100 seconds after the last tph value

# Select data within the specified time range
selected_data = df_ph[
    (df_ph['bpod_frame_times'] >= tbpod_start) &
    (df_ph['bpod_frame_times'] <= tbpod_end)
]
# Now, selected_data contains the rows of df_ph within the desired time range 
selected_data 

df_ph = selected_data

#===========================================================================
#      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
#===========================================================================
df_ph = df_ph.reset_index(drop=True)
def start_2_end_1(df_photometry): 
    """
    input = raw photometry data
    output = photometry dataframe without the initial flag=0, starting at flag=2, finishing at flag=1, reset_index applied 
    """
    df_photometry = df_photometry.reset_index(drop=True)
    array1 = df_photometry
    if array1["LedState"][0] == 0: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if (array1["LedState"][0] != 2) or (array1["LedState"][0] != 1): 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["LedState"][0] == 1: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["LedState"][len(array1)-1] == 2: 
        array1 = array1[0:len(array1)-1] 
        array1 = array1.reset_index(drop=True)
    array2 = pd.DataFrame(array1)
    return(array2) 
def start_17_end_18(df_photometry): 
    """
    input = raw photometry data
    output = photometry dataframe without the initial flag=16, starting at flag=17, finishing at flag=18, reset_index applied 
    """
    df_photometry = df_photometry.reset_index(drop=True)
    array1 = df_photometry
    if array1["Flags"][0] == 16: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["Flags"][0] == 18: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["Flags"][len(array1)-1] == 17: 
        array1 = array1[0:len(array1)-1] 
        array1 = array1.reset_index(drop=True)
    array2 = pd.DataFrame(array1)
    return(array2) 
""" 4.1.1 Change the Flags that are combined to Flags that will represent only the LED that was on """ 
"""1 and 17 are isosbestic; 2 and 18 are GCaMP"""
def change_flags(df_with_flags): 
    df_with_flags = df_with_flags.reset_index(drop=True)
    if 'LedState' in df_with_flags.columns: 
        array1 = np.array(df_with_flags["LedState"])
        for i in range(0,len(df_with_flags)): 
            if array1[i] == 529 or array1[i] == 273 or array1[i] == 785 or array1[i] == 17: 
                array1[i] = 1
            elif array1[i] == 530 or array1[i] == 274 or array1[i] == 786 or array1[i] == 18: 
                array1[i] = 2
            else: 
                array1[i] = array1[i] 
        array2 = pd.DataFrame(array1)
        df_with_flags["LedState"] = array2
        return(df_with_flags) 
    else: 
        array1 = np.array(df_with_flags["Flags"])
        for i in range(0,len(df_with_flags)): 
            if array1[i] == 529 or array1[i] == 273 or array1[i] == 785 or array1[i] == 17: 
                array1[i] = 1
            elif array1[i] == 530 or array1[i] == 274 or array1[i] == 786 or array1[i] == 18: 
                array1[i] = 2
            else: 
                array1[i] = array1[i] 
        array2 = pd.DataFrame(array1)
        df_with_flags["Flags"] = array2
        return(df_with_flags) 

def LedState_or_Flags(df_PhotometryData): 
    if 'LedState' in df_PhotometryData.columns:                         #newversion 
        df_PhotometryData = start_2_end_1(df_PhotometryData)
        df_PhotometryData = df_PhotometryData.reset_index(drop=True)
        df_PhotometryData = (change_flags(df_PhotometryData))
    else:                                                               #oldversion
        df_PhotometryData = start_17_end_18(df_PhotometryData) 
        df_PhotometryData = df_PhotometryData.reset_index(drop=True) 
        df_PhotometryData = (change_flags(df_PhotometryData))
        df_PhotometryData["LedState"] = df_PhotometryData["Flags"]
    return df_PhotometryData
def verify_length(df_PhotometryData): 
    """
    Checking if the length is different
    x = df_470
    y = df_415
    """ 
    x = df_PhotometryData[df_PhotometryData.LedState==2]
    y = df_PhotometryData[df_PhotometryData.LedState==1] 
    if len(x) == len(y): 
        print("Option 1: same length :)")
    else: 
        print("Option 2: SOMETHING IS WRONG! Different len's") 
    print("470 = ",x.LedState.count()," 415 = ",y.LedState.count())
    return(x,y)

""" 4.1.2 Check for LedState/previous Flags bugs """ 
""" 4.1.2.1 Length """
# Verify the length of the data of the 2 different LEDs
df_470, df_415 = verify_length(df_ph)
""" 4.1.2.2 Verify if there are repeated flags """ 
""" 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
df_ph_1 = df_ph
def start_2_end_1(df_photometry): 
    """
    input = raw photometry data
    output = photometry dataframe without the initial flag=0, starting at flag=2, finishing at flag=1, reset_index applied 
    """
    df_photometry = df_photometry.reset_index(drop=True)
    array1 = df_photometry
    if array1["LedState"][0] == 0: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if (array1["LedState"][0] != 2) or (array1["LedState"][0] != 1): 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["LedState"][0] == 1: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["LedState"][len(array1)-1] == 2: 
        array1 = array1[0:len(array1)-1] 
        array1 = array1.reset_index(drop=True)
    array2 = pd.DataFrame(array1)
    return(array2) 
def start_17_end_18(df_photometry): 
    """
    input = raw photometry data
    output = photometry dataframe without the initial flag=16, starting at flag=17, finishing at flag=18, reset_index applied 
    """
    df_photometry = df_photometry.reset_index(drop=True)
    array1 = df_photometry
    if array1["Flags"][0] == 16: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["Flags"][0] == 18: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["Flags"][len(array1)-1] == 17: 
        array1 = array1[0:len(array1)-1] 
        array1 = array1.reset_index(drop=True)
    array2 = pd.DataFrame(array1)
    return(array2) 
df_ph = df_ph.reset_index(drop=True)
df_ph = LedState_or_Flags(df_ph)
df_nph = df_nph[1:len(df_nph)+1].reset_index(drop=True)
df_ph_1 = df_ph

# %%
# Remove rows with LedState 1 at both ends if present
if df_ph_1['LedState'].iloc[0] == 1 and df_ph_1['LedState'].iloc[-1] == 1:
    df_ph_1 = df_ph_1.iloc[1:]

# Remove rows with LedState 2 at both ends if present
if df_ph_1['LedState'].iloc[0] == 2 and df_ph_1['LedState'].iloc[-1] == 2:
    df_ph_1 = df_ph_1.iloc[:-2]

# Filter data for LedState 2 (470nm)
df_470 = df_ph_1[df_ph_1['LedState'] == 2]

# Filter data for LedState 1 (415nm)
df_415 = df_ph_1[df_ph_1['LedState'] == 1]

# Check if the lengths of df_470 and df_415 are equal
assert len(df_470) == len(df_415), "Sync arrays are of different lengths"

# %%
# Remove rows with LedState 1 at both ends if present
if df_ph_1['LedState'].iloc[0] == 1 and df_ph_1['LedState'].iloc[-1] == 1:
    df_ph_1 = df_ph_1.iloc[1:]

# Remove rows with LedState 2 at both ends if present
if df_ph_1['LedState'].iloc[0] == 2 and df_ph_1['LedState'].iloc[-1] == 2:
    df_ph_1 = df_ph_1.iloc[:-2]

# Filter data for LedState 2 (470nm)
df_470 = df_ph_1[df_ph_1['LedState'] == 2]

# Filter data for LedState 1 (415nm)
df_415 = df_ph_1[df_ph_1['LedState'] == 1]

# Check if the lengths of df_470 and df_415 are equal
assert len(df_470) == len(df_415), "Sync arrays are of different lengths"
# %%

plt.rcParams["figure.figsize"] = (8, 5)
plt.plot(df_470[regions], c='#279F95', linewidth=0.5)
plt.plot(df_415[regions], c='#803896', linewidth=0.5)
session_info = one.eid2ref(sl.eid)
plt.title("Cropped signal "+session_info.subject+' '+str(session_info.date))
plt.legend(["GCaMP", "isosbestic"], frameon=False)
sns.despine(left=False, bottom=False)
plt.show(block=False)
plt.close() 
# Print counts
print("470 =", df_470['LedState'].count(), " 415 =", df_415['LedState'].count())







#%%
"""
KB 2025June20 
add the path to the df excel list :) 
""" 
# %%
import os
import pandas as pd

# Load the CSV
file_path_df = "/home/kceniabougrova/Downloads/KB_mice_sessions_fibers - kcenia's data reextration - fiber mappings - kcenia_export.csv"
df = pd.read_csv(file_path_df)

# Base folder to search in
base_path = "/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/kb/data/one/mainenlab/Subjects"

# Prepare result lists
updated_rows = []
repeated_entries = []

for idx, row in df.iterrows():
    subject = row['subject']
    date = str(row['date'])

    session_base_path = os.path.join(base_path, subject, date)
    if not os.path.isdir(session_base_path):
        updated_rows.append(row)
        row['path'] = None
        continue

    # Check all session subfolders (e.g. 001, 002...)
    session_folders = [f for f in os.listdir(session_base_path)
                       if os.path.isdir(os.path.join(session_base_path, f))]

    matched_paths = []
    for sess in session_folders:
        file_path = os.path.join(session_base_path, sess, "raw_photometry_data", "raw_photometry.csv")
        if os.path.isfile(file_path):
            matched_paths.append(file_path)

    if len(matched_paths) == 1:
        row['path'] = matched_paths[0]
        updated_rows.append(row)
    elif len(matched_paths) > 1:
        for path in matched_paths:
            new_row = row.copy()
            new_row['path'] = path
            repeated_entries.append(new_row)
        # still include the first in the main df for completeness
        row['path'] = matched_paths[0]
        updated_rows.append(row)
    else:
        row['path'] = None
        updated_rows.append(row)

# Create the new DataFrames
df_updated = pd.DataFrame(updated_rows)
df_repeated = pd.DataFrame(repeated_entries)

# Save or inspect
df_updated.to_csv("updated_fiber_paths.csv", index=False)
df_repeated.to_csv("repeated_fiber_paths.csv", index=False)

print("Done. Results saved to 'updated_fiber_paths.csv' and 'repeated_fiber_paths.csv'")



# %%
# Replace base path in 'path' column
df_updated['path'] = df_updated['path'].str.replace(
    "/media/kceniabougrova/Seagate Basic/IBL_server_PC_20250529/",
    "/mnt/h0/",
    regex=False
)

# Save again with updated paths
df_updated.to_csv("updated_fiber_paths.csv", index=False)

print("Paths updated and files saved again.")
