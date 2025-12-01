import sys 
from nptdms import TdmsFile
import csv

def convert_tdms_to_csv(tdms_file_path, csv_file_path):
    tdms_file = TdmsFile.read(tdms_file_path)
    calib_group = tdms_file['Calib']


    channels = [calib_group[group_name] for group_name in ['date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp']]

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        header = ['Time'] + ['date-time', 'fX', 'fY', 'fZ', 'mX', 'mY', 'mZ', 'Temp']
        csv_writer.writerow(header)

        for time, *data in zip(channels[0], *(channel for channel in channels)):
            row = [time]
            row.extend(data)
            csv_writer.writerow(row)

    print(f'Conversion complete. CSV file saved to {csv_file_path}')
    return "ok" 

# tdms_file_path = r"C:\Users\Public\CM2_Data\2025_03_13_MIEC6E1_Hex8_231_Si\2025_03_13_MIEC6E1_Hex8_231_Si.tdms"
# csv_file_path = r"C:\Users\Public\CM2_Data\2025_03_13_MIEC6E1_Hex8_231_Si\2025_03_13_MIEC6E1_Hex8_231_Si.csv"
# print("Hi")
# convert_tdms_to_csv(tdms_file_path, csv_file_path)
