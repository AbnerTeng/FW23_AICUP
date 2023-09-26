"""
head
"""
import os
import pandas as pd
from shapely.geometry import Point
from .utils import (
    load_data,
    add_twd97_coordinates_to_dataframe as add_twd97
)

class NFacilities:
    """
    Find the amount of facilities within the radius
    """
    def __init__(self, facility_path: str, target_path: str, rad: int) -> None:
        self.facility = load_data(facility_path)
        self.target = load_data(target_path)
        self.rad = rad


    def split_data(self) -> tuple:
        """
        Get the (x, y) for facility and target
        """
        facility_pos = add_twd97(self.facility)[['橫坐標', '縱坐標']]
        target_pos = self.target[['橫坐標', '縱坐標']]
        return facility_pos, target_pos


    def calculate_dist(self, facility_pos: pd.DataFrame, target_pos: pd.DataFrame):
        """
        Get the distance between each facility and the target
        -----------------------------------------------------
        Example
            ref_point: POINT(305266, 2768378)
        """
        ref_point = Point(
            target_pos.iloc[0, 0],
            target_pos.iloc[0, 1]
        )
        facility_pos['distance'] = facility_pos.apply(
            lambda row: ref_point.distance(
                Point(row['橫坐標'], row['縱坐標'])
            ),
            axis = 1
        )
        return facility_pos


    def find_n_facilities(self, facility_pos: pd.DataFrame):
        """
        Find the amount of facilities within the radius
        """
        facility_pos = facility_pos[facility_pos['distance'] <= self.rad]
        return len(facility_pos)


    def main(self) -> None:
        """
        Main function
        """
        facility_pos, target_pos = self.split_data()
        facility_pos = self.calculate_dist(facility_pos, target_pos)
        n_facilities = self.find_n_facilities(facility_pos)
        print(f"N facilities: {n_facilities}")


if __name__ == "__main__":
    nfac = NFacilities(
        f"{os.getcwd()}/data/external_data/ATM資料.csv",
        f"{os.getcwd()}/data/small_training_data.csv",
        500
    )
    nfac.main()
    