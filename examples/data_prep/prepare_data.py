import pandas as pd
import numpy as np

directory = "data"
df = pd.read_csv(f"{directory}/all_data.csv")

# Columns to keep
columns_to_keep = [
    "NID",
    "Source",
    "floors_no",
    "socio_eco",
    "ulabel",
    "IMGAVAL",
]

# Filter the DataFrame to keep only the specified columns
df_filtered = df[columns_to_keep].copy()

# set the index to 'NID'
df_filtered.set_index("NID", inplace=True)

# all values that are -999 set to NaN
df_filtered.replace([-999, "-999"], np.nan, inplace=True)

# change name of 'ulabel' to 'label'
df_filtered.rename(columns={"ulabel": "label"}, inplace=True)

# change name of 'IMGAVAL' to 'has_image'
df_filtered.rename(columns={"IMGAVAL": "has_image"}, inplace=True)

# has_image should be true or false instead of 1 or 0
df_filtered["has_image"] = df_filtered["has_image"].astype(bool)

# create new column that is true if both 'floors_no' and 'socio_eco' are not NaN
df_filtered["has_tabular"] = df_filtered[["floors_no", "socio_eco"]].notna().all(axis=1)

# remove all rows where no label is present
df_filtered = df_filtered[df_filtered["label"].notna()]

# set category for socio_eco low income zone to majority low income zone
df_filtered["socio_eco"] = df_filtered["socio_eco"].replace(
    "Low-income zone", "Majority low-income zone"
)

# save to csv
df_filtered.to_csv(f"{directory}/prepared_data.csv")
