# NM_project2
Copy from NM_project | cleaned | working from new PC

## 1. 00_movefiles_changefilename.py
  - move files to corresponding date folders



## 2. 01_get_behavior_and_photometry.py
  - pick data from list:
    - KB_sessions_insertions_map - upload.csv
    - KB_sessions_insertions_map - sessions_table(1).csv
   
  - select 1 session (by selecting the row)
 
  - load df_nph, df_bnc, df_trials
  
  - preprocessing:
    - find TTL sync events
    - synchronize df_nph times with behavior times 
    - crop the photometry session around the behavior - _modify this if you want to check the signal before or after the task_
    - clean df_nph: LedState or Flags (for GCaMP and isosbestic), check for repeated flags, check for the same lenght for both LEDs
    - create final cleaned df: df_nph with times (in the clock of the behavior PC), raw_isosbestic, and raw_calcium
    - jove2019() to preprocess the data 
   
  - add new variables
    - trial_number
   
  - PLOT
    - select event and interval to plot the PSTH
    - plot heatmap
   
  - add new variables
    - trialNumber
    - allContrasts
    - allUContrasts
    - reactionTime
   
  ### TO DO:
  - [optional] add other signal preprocessing methods - to compare



## 3. 20_poster_Oct2025.py
`📶 Use regressors to investigate which variables better explain the NM signal variability`
 - pick data from list:
    - KB_sessions_insertions_map - upload.csv
    - KB_sessions_insertions_map - sessions_table(1).csv
