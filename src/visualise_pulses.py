from utils import get_path_to_data_dir
import pandas as pd
import matplotlib.pyplot as plt



def view_indxs(use_deq=True, indxs_to_view=[0, 10000, 20000, 30000, 40000, 50000, 60000]):
    """
    Function to visualise the pulses from the physical experiment and the simulation results, so I can easily track what indexes to use for a given sim.
    """
    processed_data_dir = get_path_to_data_dir() / "processed"

    wltp_df = pd.read_csv(processed_data_dir / "MLP001_wltp_25degC_record_deq.csv")
    time_column = "deq_Elapsed Time[h]"
    
    wltp_df = pd.read_csv(processed_data_dir / "MLP001_wltp_25degC_record_shortened.csv")
    time_column = "Elapsed Time[h]"

    plt.figure(figsize=(12, 6))
    for idx in indxs_to_view:
        plt.axvline(idx, color='r', linestyle='--', label=f'Index {idx}' if idx == indxs_to_view[0] else "")

    plt.plot(wltp_df[time_column].to_numpy()*3600, wltp_df["Voltage(V)"].to_numpy(), label="Physical Experiment Voltage", color='b')
    plt.axvline
    plt.xlabel("Elapsed Time [h]")
    plt.ylabel("Voltage [V]")
    plt.title("Voltage vs Elapsed Time")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    view_indxs(use_deq=False, indxs_to_view=[0, 10000, 60000])  # set to False if you want to view the original data without deq

if __name__ == "__main__":
    main()