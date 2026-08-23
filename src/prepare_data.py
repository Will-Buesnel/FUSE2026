from utils import get_path_to_data_dir
import pandas as pd


import pandas as pd

import pandas as pd

def collapse_by_mean(df, x_col='x'):
    """
    Collapse a DataFrame with multiple rows per x-value into
    a shorter DataFrame with one row per unique x.
    
    - Numeric columns (other than x_col) are averaged.
    - Non-numeric columns are kept using the first value seen
      for that x (assumes they're constant within a group;
      change 'first' to something else if not).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing x_col and other columns.
    x_col : str
        Name of the column to group by (the repeated value).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per unique x value.
    """
    # Build aggregation rules per column
    agg_dict = {}
    for col in df.columns:
        if col == x_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'first'

    result = (
        df.groupby(x_col, as_index=False)
          .agg(agg_dict)
          .sort_values(x_col)
          .reset_index(drop=True)
    )
    return result


# Example usage
if __name__ == "__main__":
    data = {
        'x': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'y': [10, 12, 11, 20, 22, 30, 31, 29, 30],
        'z': [0.1, 0.2, 0.15, 0.5, 0.55, 0.9, 0.95, 0.85, 0.88],
        'label': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'c']
    }
    df = pd.DataFrame(data)
    print("Original:")
    print(df)

    shortened = collapse_by_mean(df, x_col='x')
    print("\nShortened:")
    print(shortened)

# Example usage
if __name__ == "__main__":
    data = {
        'x': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'y': [10, 12, 11, 20, 22, 30, 31, 29, 30]
    }
    df = pd.DataFrame(data)
    print("Original:")
    print(df)

    shortened = collapse_by_mean(df, x_col='x')
    print("\nShortened (mean per x):")
    print(shortened)



if __name__ == "__main__":
    df = pd.read_csv(get_path_to_data_dir() / "processed" / "MLP001_wltp_25degC_record_deq.csv")
    df_shortened = collapse_by_mean(df, x_col='Elapsed Time[h]')
    df_shortened.to_csv(get_path_to_data_dir() / "processed" / "MLP001_wltp_25degC_record_shortened.csv", index=False)


    # to check that the function works, we can plot the original and shortened data:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(df["Elapsed Time[h]"], df["Voltage(V)"], label="Original")
    plt.plot(df_shortened["Elapsed Time[h]"], df_shortened["Voltage(V)"], label="Shortened", linestyle="--")
    plt.xlabel("Elapsed Time [h]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.show()
     