import os
import pandas as pd

root_folder = r"C:\Users\DaryapS\WIKA\EM-CEO-ORG-Resense - Files\110_Manufacturing\Calibration\Si"
output_file = os.path.join(root_folder, "combined_kpi_data.xlsx")

wanted_top_headers = ["Sensor_Name", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]

all_dfs = []

for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".xlsx") and "sensor_kpi" in file.lower():
            full_path = os.path.join(root, file)

            try:
                # Read header row 0 and 1
                df = pd.read_excel(full_path, header=[0, 1], engine="openpyxl")

                df.columns = pd.MultiIndex.from_tuples(
                    (top if pd.notna(top) else sub.split("_")[0], sub)
                    for top, sub in df.columns
                )

                df = df[[col for col in df.columns if col[0] in wanted_top_headers]]

                all_dfs.append(df)

            except Exception as e:
                print(f"Error reading {full_path}: {e}")


if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)

    combined.columns = [f"{a}_{b}" for a, b in combined.columns]

    combined.to_excel(output_file, index=False)
    print("Saved:", output_file)
else:
    print("No files found or matching columns.")