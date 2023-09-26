"""
Get average distance of k nearest neighbors
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .add_coordinates import add_twd97_coordinates_to_dataframe as add_twd97
from .utils import load_data

class GetNearestNeighbors:
    """
    Get the average distance of k nearest neighbors
    """
    def __init__(self, facility_path: str, target_path: str, k: int, facility_name: str) -> None:
        self.k = k
        self.facility = load_data(facility_path)
        self.target = load_data(target_path)
        self.facility_name = facility_name


    def split_data(self) -> pd.DataFrame:
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
        ) -> None:
        """
        get the k nearest neighbors for each buildings
        """
        facility_pos, target_pos = np.array(facility_pos), np.array(target_pos)
        nbrs = NearestNeighbors(
            n_neighbors = k,
        )
        nbrs.fit(facility_pos)
        distances, _ = nbrs.kneighbors(target_pos, n_neighbors = k)
        avg_distances = np.mean(distances, axis = 1)
        return avg_distances


    def main(self) -> None:
        """
        Main function
        """
        facility_pos, target_pos = self.split_data()
        avg_distances = self.get_avg_distances(facility_pos, target_pos, self.k)
        self.target[f'avg_distances_{self.facility_name}'] = avg_distances
        print(self.target.head())


if __name__ == "__main__":
    for external_datas in os.listdir(f"{os.getcwd()}/data/external_data"):
        get_nn = GetNearestNeighbors(
            f"{os.getcwd()}/data/external_data/{external_datas}",
            f"{os.getcwd()}/data/training_data.csv",
            3, f'{external_datas[:2]}'
        )
        print(get_nn.target.shape)
        get_nn.main()
