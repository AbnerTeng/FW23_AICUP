import os
import pandas as pd
from .knn_test import GetNearestNeighbors
from .n_facilities_v2 import NFacilities

# def merge()




if __name__ == "__main__":
    TARGET_PATH = f"{os.getcwd()}/data/training_data.csv"
    # get_nn = GetNearestNeighbors(
    #     f"{os.getcwd()}/data/lib_xy.csv",
    #     TARGET_PATH,
    #     3, 'library'
    # )
    # get_nn.main()
    nfac = NFacilities(
        f"{os.getcwd()}/data/lib_xy.csv",
        TARGET_PATH,
        2000
    )
    print(nfac.main())
    ## TODO: merge the data
    