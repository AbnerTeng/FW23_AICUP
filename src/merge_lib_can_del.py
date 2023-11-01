"""
Merge library data with training data
"""
import os
import pandas as pd
from .n_facilities_v2 import NFacilities

def merge(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two dataframe
    """
    df1['N_lib_2000'] = df2['N_facilities']
    return df1


if __name__ == "__main__":
    TARGET_PATH = f"{os.getcwd()}/data/training_data.csv"
    nfac = NFacilities(
        f"{os.getcwd()}/data/lib_xy.csv",
        TARGET_PATH,
        2000
    )
    library_target = nfac.main()
    training_data = pd.read_csv(TARGET_PATH)
    training_data = merge(training_data, library_target)
    training_data.to_csv(f"{os.getcwd()}/data/training_data.csv", index = False)    
    ## TODO: merge the data