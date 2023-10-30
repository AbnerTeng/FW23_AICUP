"""
Calculate mean distance of k nearest neighbors
"""
import os
import pandas as pd
from shapely.geometry import Point
from .utils import load_data, add_twd97_coordinates_to_dataframe as add_twd97

class MeanDist():
    """
    The class that can get the mean value of k nearest neighbors
    """
    def __init__(self, facility_path: str, target_path: str):
        self.facility = load_data(facility_path)
        self.target = load_data(target_path)
        self.facilities_with_dist, self.target_pos = self.split_data()

    def split_data(self) -> tuple:
        """
        Get the (x, y) for facility and target
        """
        facility_pos = add_twd97(self.facility)[['橫坐標', '縱坐標']]
        target_pos = self.target[['橫坐標', '縱坐標']]
        return facility_pos, target_pos


    def calc_nn_mean_dist(self,x:int, y:int, k=3)->float:
        """
        Get the mean value of point to k nearest neighbors
        """
        ref_point = Point(x, y)

        #Get the distance between each facility and the target
        self.facilities_with_dist['distance'] = self.facilities_with_dist.apply(
            lambda row: ref_point.distance(
                Point(row['橫坐標'], row['縱坐標'])
            ),
            axis = 1
        )

        #Sort the dataframe by distance
        self.facilities_with_dist = self.facilities_with_dist.sort_values(
            by = 'distance',
            ascending=True
        )

        #return mean value of k nearest neighbors
        mean = self.facilities_with_dist.iloc[0:k, 2].mean()
        return mean


    def update_dataframe(self, column_name="nn_mean_distance") -> pd.DataFrame:
        """
        Update the dataframe
        """
        # Apply the mean function using lambda and assign the result to a new column in self.target
        self.target[column_name] = self.target_pos.apply(self.__mean_function, axis=1)

        return self.target


    def __mean_function(self, row):
        """
        Private function for calculate mean
        """
        return self.calc_nn_mean_dist(row.iloc[0], row.iloc[1])

if __name__ == "__main__":
    #Just change this line before you run this script
    TARGET_PATH = f"{os.getcwd()}/data/training_data.csv"

    for external_datas in os.listdir(f"{os.getcwd()}/data/external_data"):
        #get file name
        file_name = external_datas.split('.')[0].replace('資料', '')

        print(f"{os.getcwd()}/data/external_data/{external_datas}")
        md = MeanDist(
            f"{os.getcwd()}/data/external_data/{external_datas}",
            TARGET_PATH
            )

        md.update_dataframe(column_name=f"mean_distance_to_{file_name}").to_csv(
            f"{os.getcwd()}/data/small_training_data.csv", 
            index=False
            )
