"""
KB - June2025
edit the fiber regions
""" 

#%%
import pandas as pd

# File path with correct escaping
file_path_df = "/home/kceniabougrova/Downloads/kcenia's data reextration - fiber mappings - kcenia_export.csv"
file_path_probes = "/home/kceniabougrova/Downloads/kcenia fiber insertions - insertions.csv"

# Load the CSV into a DataFrame
df = pd.read_csv(file_path_df)
df_probes = pd.read_csv(file_path_probes)

# Display the first few rows to check if it's loaded correctly
df

# %%
""" edit the "fiber" column from df with the mice with 1 fiber implant"""
import pandas as pd

# List of subjects to match
subjects_to_match = [
    'ZFM-02128', 'ZFM-03059', 'ZFM-03062', 'ZFM-03065',
    'ZFM-03447', 'ZFM-03448', 'ZFM-04026', 'ZFM-04019',
    'ZFM-04533', 'ZFM-05235', 'ZFM-05236', 'ZFM-05245',
    'ZFM-05248', 'ZFM-05645', 'ZFM-06268', 'ZFM-06271',
    'ZFM-06171', 'ZFM-06946', 'ZFM-06948'
]

# Ensure 'subject' column is treated as a string in both dataframes
df['subject'] = df['subject'].astype(str)
df_probes['subject'] = df_probes['subject'].astype(str)

# Filter the df_probes dataframe to keep only the relevant subjects
df_probes_filtered = df_probes[df_probes['subject'].isin(subjects_to_match)]

# Merge df with df_probes_filtered on 'subject', keeping the 'probename' column
df = pd.merge(df, df_probes_filtered[['subject', 'probename']], on='subject', how='left')

# Update the 'fiber' column in df with the corresponding 'probename' values from df_probes
df['fiber'] = df['probename'].where(df['probename'].notnull(), df['fiber'])

# Drop the 'probename' column as it's no longer needed
df = df.drop(columns=['probename'])

# Display the updated dataframe
df

# %%
""" count the number of NaNs in df "fiber" column """
# Count the number of NaN values in the 'fiber' column
nan_count = df['fiber'].isna().sum()

# Display the result
print(f'Number of NaN values in the fiber column: {nan_count}')

# %%
# Example with your path:
df.to_csv('/home/kceniabougrova/Downloads/updated_data.csv', index=False)

# %%
