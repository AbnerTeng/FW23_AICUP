import geopandas as gpd
import pandas as pd

def add_social_economic_feature(house: pd.DataFrame) -> pd.DataFrame:
    """
    This is a function to add 3 social economic village features into training/testing data.
    "social_economic_feature.shp" contains 3 features "avg_tax", "density", "edu_p".
    Note: house should contain "橫坐標" & "縱坐標".
    """
    village_feature = gpd.read_file("../data/new_data/social_economic_data/social_economic_feature.shp")
    house_gdf = gpd.GeoDataFrame(house, geometry = gpd.points_from_xy(house["橫坐標"], house["縱坐標"]), crs = "EPSG:3826")
    house_gdf = gpd.sjoin(house_gdf, village_feature, how = "left")
    return pd.DataFrame(house_gdf.drop(columns=["V_ID", "geometry", "index_right"]))