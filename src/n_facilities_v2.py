"""
files for calculate the amount of facilities within the radius
Author: Guan-Yu, Lin / Yu-Chen, Den
--------------------
Argumemts:
    --radius: radius of the circle

Example:
    python -m src.n_facilities_v2 --radius 500
"""
import os
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
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


    def calculate_dist(
            self, facility_pos: pd.DataFrame, target_pos: pd.DataFrame, num: int
        ) -> pd.DataFrame:
        """
        Get the distance between each facility and the target
        -----------------------------------------------------
        Example
            ref_point: POINT(305266, 2768378)
        """
        ref_point = Point(
            target_pos.iloc[num, 0],
            target_pos.iloc[num, 1]
        )
        facility_pos['distance'] = facility_pos.apply(
            lambda row: ref_point.distance(
                Point(row['橫坐標'], row['縱坐標'])
            ),
            axis = 1
        )
        return facility_pos


    def find_n_facilities(self, facility_pos: pd.DataFrame) -> int:
        """
        Find the amount of facilities within the radius
        """
        facility_pos = facility_pos[facility_pos['distance'] <= self.rad]
        return len(facility_pos)


    def main(self) -> pd.DataFrame:
        """
        Main function
        """
        target_calculated = self.target.copy()
        target_calculated['N_facilities'] = 0
        target_calculated['N_facilities'] = target_calculated['N_facilities'].astype(int)
        facility_pos, target_pos = self.split_data()

        for i in tqdm(range(len(target_pos))):
            facility_pos = self.calculate_dist(facility_pos, target_pos, i)
            n_facilities = self.find_n_facilities(facility_pos)
            target_calculated.loc[i, 'N_facilities'] = n_facilities

        target_calculated = target_calculated[['ID', 'N_facilities']]
        return target_calculated
