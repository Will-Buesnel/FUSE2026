"""Will Buesnel, Jul 26."""

from utils import dequantise_data, get_drive_cycle_hours_col
import pandas as pd
from pathlib import Path

def main():
    root = Path.cwd().resolve()

    processed_data_dir = root / "data" / "processed"
    undeq_dataset_path = processed_data_dir / "MLP001_wltp_25degC_record.csv"
    # process the hours
    undeq_dataset = get_drive_cycle_hours_col(undeq_dataset_path)
    dequantised_time = dequantise_data(undeq_dataset["Elapsed Time[h]"].to_numpy() * 3600, pbar=True)
    # rewrite the df with undeq columns.
    dequantised_df = undeq_dataset.copy()
    dequantised_df["deq_Elapsed Time[h]"] = dequantised_time / 3600

    # drop duplicate rows at the end of the dataset, which are caused by the dequantisation process.
    dequantised_df = dequantised_df.drop_duplicates(subset=["deq_Elapsed Time[h]"], keep="first")
    # write to csv
    dequantised_df.to_csv(processed_data_dir / "MLP001_wltp_25degC_record_deq.csv", index=False)
    print("Dequantised dataset saved to data/processed/MLP001_wltp_25degC_record_deq.csv")

if "__main__" == __name__:
    main()