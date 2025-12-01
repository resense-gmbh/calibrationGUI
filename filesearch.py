
import os
import pandas as pd

onedrive_path = r"C:\Users\DaryapS\WIKA\EM-CEO-ORG-Resense - Files\110_Manufacturing\Calibration"
columns_to_keep = ['Sensor_Name', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']  # Adjust to your KPIs
dataframes = []

for root, dirs, files in os.walk(onedrive_path):
    for file in files:
        if file.lower().endswith(".xlsx"):
            full_path = os.path.join(root, file)
            try:
                # Read Excel with three header rows
                df = pd.read_excel(full_path, header=[0,1], engine='openpyxl')
                df.dropna(how='all', inplace=True)

                # Flatten MultiIndex columns
                df.columns = [
                    f"{top}_{sub}".strip("_").replace(" ", "")
                    for top, sub in df.columns
                ]
                
                # Filter columns
                filtered_cols = [col for col in columns_to_keep if col in df.columns]
                if filtered_cols:
                    df = df[filtered_cols]
                    dataframes.append(df)
            except Exception as e:
                print(f"Could not read {full_path}: {e}")

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    output_file = os.path.join(onedrive_path, "combined_kpi_data.xlsx")
    combined_df.to_excel(output_file, index=False)
    print(f"Saved filtered KPI data to {output_file}")
else:
    print("No matching columns found.")
