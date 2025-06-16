"""
KB - June2025
edit the fiber regions
""" 

#%%
import pandas as pd

""" load data frames """
file_path_df = "/home/kceniabougrova/Downloads/kcenia's data reextration - fiber mappings - kcenia_export.csv"
file_path_probes = "/home/kceniabougrova/Downloads/kcenia fiber insertions - insertions.csv"

df = pd.read_csv(file_path_df)
df_probes = pd.read_csv(file_path_probes)
df

# %%
""" DONE AND SAVED IN df """
# #===========================================================================
# """ edit the "fiber" column from df with the mice with 1 fiber implant """
# subjects_to_match = [
#     'ZFM-02128', 'ZFM-03059', 'ZFM-03062', 'ZFM-03065',
#     'ZFM-03447', 'ZFM-03448', 'ZFM-04026', 'ZFM-04019',
#     'ZFM-04533', 'ZFM-05235', 'ZFM-05236', 'ZFM-05245',
#     'ZFM-05248', 'ZFM-05645', 'ZFM-06268', 'ZFM-06271',
#     'ZFM-06171', 'ZFM-06946', 'ZFM-06948'
# ]

# df['subject'] = df['subject'].astype(str)
# df_probes['subject'] = df_probes['subject'].astype(str)

# # Filter the df_probes dataframe to keep only the relevant subjects
# df_probes_filtered = df_probes[df_probes['subject'].isin(subjects_to_match)]

# # Merge df with df_probes_filtered on 'subject', keeping the 'probename' column
# df = pd.merge(df, df_probes_filtered[['subject', 'probename']], on='subject', how='left')

# # Update the 'fiber' column in df with the corresponding 'probename' values from df_probes
# df['fiber'] = df['probename'].where(df['probename'].notnull(), df['fiber'])

# # Drop the 'probename' column as it's no longer needed
# df = df.drop(columns=['probename'])

# # Display the updated dataframe
# df
# # df.to_csv(file_path_df, index=False)


# %%
""" count the number of NaNs in df "fiber" column """
nan_count = df['fiber'].isna().sum()
print(f'Number of NaN values in the fiber column: {nan_count}')

#%%
""" Check the """
# Load the Excel file into a DataFrame
df_excel = pd.read_excel('/home/kceniabougrova/Downloads/saved_all_sheet_df_eid (1).xlsx') 
df_excel["subject"] = df_excel["mouse"]

# Make sure 'subject' and 'date' columns are strings for proper comparison
df['subject'] = df['subject'].astype(str)
df['date'] = df['date'].astype(str)
df_excel['subject'] = df_excel['subject'].astype(str)
df_excel['date'] = df_excel['date'].astype(str)

# Merge to find non-overlapping (subject, date) pairs from df_excel not in df
merged_check = pd.merge(
    df_excel,
    df[['subject', 'date']],
    on=['subject', 'date'],
    how='left',
    indicator=True
)

# Filter only the rows that are in df_excel but not in df
df_new = merged_check[merged_check['_merge'] == 'left_only'].drop(columns=['_merge'])

# Optional: reset index
df_new = df_new.reset_index(drop=True)

# Display how many new rows were found
print(f"{len(df_new)} new (subject, date) pairs found in df_excel but not in df.")
#%%








































