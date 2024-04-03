"""
useful utils
"""
from typing import List, Tuple, Union
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor


def load_data(path: str) -> Union[pd.DataFrame, dict]:
    """
    load .csv files
    """
    if path.split(".")[-1] == "csv":
        data = pd.read_csv(path, encoding='utf-8')

    elif path.split(".")[-1] == "yaml":
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError("File type not supported.")

    return data


def logarithm(data: pd.DataFrame) -> pd.DataFrame:
    """
    log transformation function

    Args:
        data (pd.DataFrame): features

    Returns:
        pd.DataFrame: features with log transformation
    """
    return data.apply(lambda x: np.log1p(x))


def train_test_split(
        feat: pd.DataFrame,
        label: pd.Series,
        ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets

    Args:
        feat (pd.DataFrame): features
        label (pd.Series): labels
        ratio (float): train / test ratio

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    """
    x_train, x_valid = \
        feat[:int(len(feat) * ratio)], feat[int(len(feat) * ratio):]

    y_train, y_valid = \
        label[:int(len(label) * ratio)], label[int(len(label) * ratio):]
    return x_train, x_valid, y_train, y_valid



def one_hot_encoding(data: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """
    One-hot-encoding function
    """
    data_encoded = pd.get_dummies(data, columns=cat_cols)
    return data_encoded


def add_coordinates(data: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    function to add coordinates into a pandas DataFrame.
    
    method: str -> "twd97", "wgs84"
    """
    new = "twd97" if method == "wgs84" else "wgs84"
    _map = {
        'twd97': ["EPSG:4326", "lng", "lat"],
        'wgs84': ["EPSG:3826", "橫坐標", "縱坐標"],
    }
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(
            data[_map[method][1]], data[_map[method][2]]
        ),
        crs=_map[method][0]
    ).to_crs(
        epsg=int(_map[new][0].split(":")[1])
    )
    new_coords = gdf.get_coordinates().rename(
        columns={"x": _map[new][1], "y": _map[new][2]}
    )
    return pd.concat([data, new_coords], axis=1)


def feature_select(
        data: pd.DataFrame,
        pred_target: str,
        dims: int,
        model=RandomForestRegressor,
    ) -> Tuple[pd.DataFrame, pd.Series]:
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
    features, output = data.drop(columns=[pred_target]), data[pred_target]
    model.fit(features, output)
    feature_importance = model.feature_importances_
    print(feature_importance)
    indices = np.argsort(feature_importance)[::-1][:dims]
    dim_reduction_data = pd.DataFrame()

    for idx in indices:
        dim_reduction_data[features.columns[idx]] = features[features.columns[idx]]
    return dim_reduction_data, output
    