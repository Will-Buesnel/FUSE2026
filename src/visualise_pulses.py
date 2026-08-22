from utils import get_path_to_data_dir
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the experimental data
    exp_data_path = get_path_to_data_dir() / "processed" / "MLP001_wltp_25degC_record_deq.csv"
    experiment_df = pd.read_csv(exp_data_path)
    idxes = [ 334000, 344000]  # specify the indices where you want to draw vertical lines
    plt.figure(figsize=(12, 6))
    plt.plot(experiment_df["deq_Elapsed Time[h]"], experiment_df["Voltage(V)"], label="Voltage")
    for index in idxes:
        plt.axvline(x=experiment_df["deq_Elapsed Time[h]"].iloc[index], color='r', linestyle='--', label=f'Index {index}')
    plt.xlabel("Elapsed Time [h]")
    plt.ylabel("Voltage [V]")
    plt.title("Voltage vs Elapsed Time with Vertical Lines at Specific Indices")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()