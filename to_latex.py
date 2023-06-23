import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load your CSV, considering the first row as headers
def to_latex(dir):
    df = pd.read_csv(dir, skipinitialspace=True)

    # Delete the first column
    print(df.iloc[1, 0])
    df = df.drop(df.columns[0], axis=1)
    # df = df.iloc[:, :-1].drop("training_accuracy", axis=1)


    # Convert the columns to a suitable numeric type
    for col in df.columns:
        # Split the column on any character that isn't a digit or a dot
        split_data = df[col].astype(str).str.split('[^0-9.]', expand=True)
        # Stack the split data into a single column
        stacked_data = split_data.stack()
        # Convert the data to floats
        stacked_data = pd.to_numeric(stacked_data, errors='coerce')
        # Unstack the data to get it back into the original shape
        df[col] = stacked_data.unstack()

    # Compute mean and std
    mean = df.mean()
    std = df.std()


    # Create new DataFrame with the format mean(std)
    summary_df = pd.DataFrame({col: f"{mean[col]:.2f} ({std[col]:.2f})" for col in df.columns}, index=[0])

    # Output as LaTeX
    latex_output = summary_df.to_latex(index=False, escape=False)
    latex_output = latex_output.split('\n')[4]
    print(latex_output)

import os
for dir in os.listdir("."):
    if (
        dir.endswith("csv")
        and "False" in dir
        and "multi_" in dir
        and "iris" in dir):
        print("dir: ", dir)
        to_latex(dir)