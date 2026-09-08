"""
Will Buesnel, Jul 26.
Turn excel files into csv.
"""

from pathlib import Path

import pandas as pd
import numpy as np

def save_excel_file_as_csv(excel_file_path: str, csv_file_path: str, sheet_name: str = "record"):
    """
    Save an Excel file as a CSV file.
    """
    df = pd.read_excel(excel_file_path, engine='openpyxl', sheet_name=sheet_name)
    df.to_csv(csv_file_path, index=False)


def convert_datetime_to_hours(df: pd.DataFrame, time_col: str = "Total Time") -> pd.DataFrame:
    hours_arr = df[time_col].str.split(":") # gives a 2d array of nx3
    arr = np.asarray(hours_arr.tolist(), dtype=float)
    hours = arr @ np.array([1, 1/60, 1/3600])

    # take away the initial time to get elapsed time.
    hours = hours - hours[0]
    return hours
    
def main():
    root = Path.cwd().resolve()
    raw_data_dir = root / "data" / "raw"
    processed_data_dir = root / "data" / "processed"

    # convert the excel file to csv
    excel_file_path = raw_data_dir / "MLP001_wltp_25degC.xlsx"
    csv_file_path = processed_data_dir / "MLP001_wltp_25degC_record.csv"
    save_excel_file_as_csv(excel_file_path, csv_file_path)


if __name__ == "__main__":
    main()