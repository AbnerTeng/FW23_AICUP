"""
File for calculate the mean value of k nearest neighbors

Author: Tzu-Hao, Liu / Yu-Chen, Den
-----------------------------------
Arguments:
    --k: k nearest neighbors

Example:
    python -m src.mean_dist --k 3
"""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Point
from ..utils.data_utils import (
    load_data,
    add_coordinates
)


class MeanDist:
    """
    The class that can get the mean value of k nearest neighbors
    """
    def __init__(self, facility_path: str, target_path: str, k: int, facility_name: str) -> None:
        self.facility = load_data(facility_path)
        self.target = load_data(target_path)
        self.facilities_with_dist, self.target_pos = self.split_data()
        self.facility_name = facility_name
        self.k = k


    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the (x, y) for facility and target
        """
        facility_pos = add_coordinates(self.facility, method="twd97")[['橫坐標', '縱坐標']]
        target_pos = self.target[['橫坐標', '縱坐標']]
        return facility_pos, target_pos


    def get_avg_distances(
        self,
        facility_pos: pd.DataFrame,
        target_pos: pd.DataFrame,
        k: int
    ) -> np.ndarray:
        """
        get the k nearest neighbors for each buildings
        """
        facility_pos, target_pos = np.array(facility_pos), np.array(target_pos)
        nbrs = NearestNeighbors(
            n_neighbors=k,
        )
        nbrs.fit(facility_pos)
        distances, _ = nbrs.kneighbors(target_pos, n_neighbors=k)
        return distances


    def calc_nn_mean_dist(self, x: int, y: int) -> float:
        """
        Get the mean value of point to k nearest neighbors
        """
        ref_point = Point(x, y)

        self.facilities_with_dist['distance'] = self.facilities_with_dist.apply(
            lambda row: ref_point.distance(
                Point(row['橫坐標'], row['縱坐標'])
            ),
            axis = 1
        )

        self.facilities_with_dist = self.facilities_with_dist.sort_values(
            by='distance',
            ascending=True
        )

        mean = self.facilities_with_dist.iloc[0: self.k, 2].mean()
        return mean


    def update_dataframe(self, column_name: str="nn_mean_distance") -> pd.DataFrame:
        """
        Update the dataframe
        """
        self.target[column_name] = self.target_pos.apply(self.__mean_function, axis=1)

        return self.target


    def __mean_function(self, row: pd.Series) -> float:
        """
        Private function for calculate mean
        """
        return self.calc_nn_mean_dist(row.iloc[0], row.iloc[1])


    def main_knn(self) -> None:
        """
        Main function
        """
        distances = self.get_avg_distances(self.facilities_with_dist, self.target_pos, self.k)
        avg_distances = np.mean(distances, axis = 1)
        self.target[f'avg_distances_{self.facility_name}'] = avg_distances
        print(self.target.head())
