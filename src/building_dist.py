"""
Calculate the mean distance between buildings and certain facilities
"""

import pandas as pd
from .add_coordinates import add_twd97_coordinates_to_dataframe
from .utils import load_data

def calculate_distance(x_1: float, y_1: float, x_2: float, y_2: float) -> float:
    """
    calculate 2 dimensional distance between two points
    """
    return ((x_1-x_2)**2 + (y_1-y_2)**2)**0.5

def find_nearest_n_facilities(x: float, y: float, facility_data: pd.DataFrame(), k: int) -> list:
    """
    find the nearest k facilities from the target point
    x: x coordinate of the target point
    y: y coordinate of the target point
    facility_data: the dataframe of the facilities(TWD97)
    k: the number of nearest facilities
    """
    tmp = []

    #for each facility, calculate the distance between the facility and the target point
    for i in range(len(facility_data)):
        tmp.append({
            "coord": (facility_data["橫坐標"][i], facility_data["縱坐標"][i]),
            "distance": calculate_distance(x, y, facility_data["橫坐標"][i], facility_data["縱坐標"][i])
        })

    #sort by distance
    tmp = sorted(tmp, key=lambda x: x["distance"])

    #return the nearest k facilities
    return tmp[:k]

def calculate_mean_distance(data: pd.DataFrame(), facility_data: pd.DataFrame(),\
                            k: int, column_name: str):
    """
    calculate the mean distance between each building and its nearest k facilities
    data: the dataframe of the training data(TWD97)
    facility_data: the dataframe of the facilities
    k: the number of nearest facilities
    """

    for i in range(len(data)):
        nearest_facilities = find_nearest_n_facilities(data["橫坐標"][i], data["縱坐標"][i],\
                                                       facility_data, k)
        data.loc[i, column_name] = sum([x["distance"] for x in nearest_facilities])/k

    print(data)
    return data

if __name__ == "__main__":
    facilities_data = load_data("data/external_data/捷運站點資料.csv")
    facilities_data = add_twd97_coordinates_to_dataframe(facilities_data)

    target_data = load_data("data/small_training_data.csv")
    #save to csv
    calculate_mean_distance(target_data, facilities_data, 3, column_name="nearest_MRT_distance").\
        to_csv("data/training_data_with_nearest_3_distance.csv", index=False)
 