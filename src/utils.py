"""
useful utils
"""
import numpy as np
import pandas as pd
import geopandas as gpd
# from sklearn.ensemble import RandomForestRegressor

def load_data(path: str) -> pd.DataFrame:
    """
    load .csv files
    """
    data = pd.read_csv(path, encoding = 'utf-8')
    return data


def one_hot_encoding(data: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    One-hot-encoding function
    """
    data_encoded = pd.get_dummies(data, columns = cat_cols)
    return data_encoded


def add_twd97_coordinates_to_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    This is a function to add TWD97 coordinates into a pandas DataFrame which contains WGS84 coordinates.
    Note: original column names of coordinates are lng & lat, new column names of coordinates are 橫坐標 & 縱坐標
    """

    gdf = gpd.GeoDataFrame(
        data,
        geometry = gpd.points_from_xy(data.lng, data.lat),
        crs = "EPSG:4326"
    ).to_crs(epsg = 3826)
    new_coords = gdf.get_coordinates().rename(columns={"x": "橫坐標", "y": "縱坐標"})
    return pd.concat([data, new_coords], axis=1)


def add_wgs84_coordinates_to_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    This is a function to add WGS84 coordinates into a pandas DataFrame which contains TWD97 coordinates.
    Note: original column names of coordinates are 橫坐標 & 縱坐標, new column names of coordinates are lng & lat
    """

    gdf = gpd.GeoDataFrame(
        data,
        geometry = gpd.points_from_xy(data.橫坐標, data.縱坐標),
        crs = "EPSG:3826"
    ).to_crs(epsg = 4326)
    new_coords = gdf.get_coordinates().rename(columns={"x": "lng", "y": "lat"})
    return pd.concat([data, new_coords], axis=1)


def feature_select(
        data: pd.DataFrame,
        pred_target: str,
        dims: int,
        # model=RandomForestRegressor,
    ) -> tuple:
    """
    Feature selection function
    
    Warnings:
        Remember to drop the useless columns before using this function.
    
    Arguments
    ---------
    - data: pd.DataFrame
        The data you want to select features from.
    - pred_target: str
        The y label of the data.
    - dims: int
        The number of important features you want to select.
    - model: sklearn model
        The model you want to use to select features.
        default: RandomForestRegressor
    """
    features, output = data.drop(columns = [pred_target]), data[pred_target]
    model.fit(features, output)
    feature_importance = model.feature_importances_
    print(feature_importance)
    indices = np.argsort(feature_importance)[::-1][:dims]
    dim_reduction_data = pd.DataFrame()

    for idx in indices:
        dim_reduction_data[features.columns[idx]] = features[features.columns[idx]]
    return dim_reduction_data, output
    