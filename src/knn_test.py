"""
Get average distance of k nearest neighbors
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .utils import (
    load_data,
    add_twd97_coordinates_to_dataframe as add_twd97
)

class GetNearestNeighbors:
    """
    Get the average distance of k nearest neighbors
    """
    def __init__(self, facility_path: str, target_path: str, k: int, facility_name: str) -> None:
        self.k = k
        self.facility = load_data(facility_path)
        self.target = load_data(target_path)
        self.facility_name = facility_name


    def split_data(self) -> tuple:
        """
        Get the (x, y) for facility and target
        """
        facility_pos = add_twd97(self.facility)[['橫坐標', '縱坐標']]
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
            n_neighbors = k,
        )
        nbrs.fit(facility_pos)
        distances, _ = nbrs.kneighbors(target_pos, n_neighbors = k)
        return distances


    def update_columns(self, avg_dist):
        """
        Update the dataframe
        """
        self.target[f'avg_distances_{self.facility_name}'] = avg_dist
        return self.target


    def main(self) -> np.ndarray:
        """
        Main function
        """
        facility_pos, target_pos = self.split_data()
        facility_pos.fillna(0, inplace=True)
        distances = self.get_avg_distances(facility_pos, target_pos, self.k)
        avg_distances = np.mean(distances, axis = 1)
        return avg_distances
