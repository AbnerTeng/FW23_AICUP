import pandas as pd
import geopandas as gpd

def add_twd97_coordinates_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    This is a function to add TWD97 coordinates into a pandas DataFrame which contains WGS84 coordinates.
    Note: original column names of coordinates are lng & lat, new column names of coordinates are 橫坐標 & 縱坐標
    """
    
    gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.lng, df.lat), crs = "EPSG:4326").to_crs(epsg = 3826)
    new_coords = gdf.get_coordinates().rename(columns={"x": "橫坐標", "y": "縱坐標"})
    return pd.concat([df, new_coords], axis=1)

def add_wgs84_coordinates_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    This is a function to add WGS84 coordinates into a pandas DataFrame which contains TWD97 coordinates.
    Note: original column names of coordinates are 橫坐標 & 縱坐標, new column names of coordinates are lng & lat
    """
    
    gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.橫坐標, df.縱坐標), crs = "EPSG:3826").to_crs(epsg = 4326)
    new_coords = gdf.get_coordinates().rename(columns={"x": "lng", "y": "lat"})
    return pd.concat([df, new_coords], axis=1)