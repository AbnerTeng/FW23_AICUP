"""
File for preprocessing educational institution data
"""
import os
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point
from ..utils.data_utils import (
    load_data,
    add_coordinates
)

PATH = f'{os.getcwd()}/data/external_data'
PATH_traindata = f'{os.getcwd()}/data/training_data.csv'


class PreprocessingEdu:
    """
    class of educational data preprocessing
    """
    @staticmethod
    def general_preprocess(file_name: str, add_coor: bool) -> pd.DataFrame:
        """
        General preprocessing for educational data
        """
        data_path = os.path.join(PATH, file_name)
        data = load_data(data_path)
        if add_coor:
            data = add_coordinates(data, "twd97")
        return data


    def preprocess_univ(self) -> pd.DataFrame:
        """
        Preprocess university data
        """
        data = self.general_preprocess(
            '大專校院基本資料.csv', add_coor=False
        )
        data = data.groupby(
            '學校名稱', as_index=False
        ).agg(
            {
                '縣市名稱': 'first',
                '總計': 'sum',
                'lat': 'first',
                'lng': 'first'
            }
        )
        data = data.rename(columns={'總計': '學生總數'})
        data['縣市名稱'] = data['縣市名稱'].str.replace(r'\d| ', '', regex=True)
        data = add_coordinates(data, "twd97")

        return data


    def preprocessing_shs(self) -> pd.DataFrame:
        """
        Preprocess senior high school data
        """
        data = self.general_preprocess(
            '高中基本資料.csv', add_coor=True
        )
        data['教師總數'] = data['專任教師數男'] + data['專任教師數女'] + data['兼任教師數男'] + data['兼任教師數女']
        data['學生總數'] = data['學生數男'] + data['學生數女']
        data = data[['學校名稱', '縣市名稱', '教師總數', '學生總數', 'lat', 'lng', '橫坐標', '縱坐標']]

        return data


    def preprocessing_jhs(self) -> pd.DataFrame:
        """
        Preprocess junior high school data
        """
        data = self.general_preprocess(
            '國中基本資料_v2.csv', add_coor=True
        )

        data['教師總數'] = data['男專任教師'] + data['女專任教師']
        data['學生總數'] = data['學生數7年級男'] + data['學生數7年級女'] + data['學生數8年級男'] +\
                        data['學生數8年級女'] + data['學生數8年級男'] + data['學生數8年級女']

        data = data[
            [
                '學校名稱', '縣市名稱', '教師總數', '學生總數',
                'Is_Combined', 'Is_Popular', 'lat', 'lng',
                '橫坐標', '縱坐標'
            ]
        ]
        return data


    def preprocessing_es(self) -> pd.DataFrame:
        """
        Preprocess junior high school data
        """
        data = self.general_preprocess(
            '國小基本資料_v2.csv', add_coor=True
        )
        data['教師總數'] = data['男專任教師'] + data['女專任教師']

        data['學生總數'] = 0
        for i in range(1, 7, 1):
            data['學生總數'] += data[f'{i}年級男學生數']
            data['學生總數'] += data[f'{i}年級女學生數']

        data = data[['學校名稱', '縣市名稱', '教師總數', '學生總數', 'Is_Popular', 'lat', 'lng', '橫坐標', '縱坐標']]

        return data


    def merge_es_info(self, training_data:pd.DataFrame, es_data:pd.DataFrame):
        """
        pass
        """
        column_names = training_data.columns

        for row in tqdm(training_data.itertuples()):
            unit_x = row[column_names.get_loc('橫坐標') + 1]
            unit_y = row[column_names.get_loc('縱坐標') + 1]

            nearest_index = self.find_nearest_facility(unit_x, unit_y, es_data)
            training_data['鄰近熱門國小'] = es_data['Is_Popular'].iloc[nearest_index]
        return training_data


    def merge_jhs_info(self, training_data:pd.DataFrame, jhs_data:pd.DataFrame):
        """
        pass
        """
        column_names = training_data.columns

        for row in tqdm(training_data.itertuples()):
            unit_x = row[column_names.get_loc('橫坐標') + 1]
            unit_y = row[column_names.get_loc('縱坐標') + 1]

            nearest_index = self.find_nearest_facility(unit_x, unit_y, jhs_data)
            training_data['鄰近熱門國中'] = jhs_data['Is_Popular'].iloc[nearest_index]
            training_data['鄰近完全中學'] = jhs_data['Is_Combined'].iloc[nearest_index]

        return training_data


    def find_nearest_facility(self, unit_x: float, unit_y: float, facilities_with_dist) -> float:
        """
        Get the index of the nearest neighbor
        """
        ref_point = Point(unit_x, unit_y)

        facilities_with_dist['distance'] = facilities_with_dist.apply(
            lambda row: ref_point.distance(
                Point(row['橫坐標'], row['縱坐標'])
            ),
            axis = 1
        )

        facilities_with_dist = facilities_with_dist.sort_values(
            by = 'distance',
            ascending=True
        )

        nearest_index = facilities_with_dist.head(1).index.tolist()[0]
        return nearest_index


    def create_edu_feature(self) -> None:
        """
        create educational features
        """
        trainingdata = load_data(PATH_traindata)
        es_processing_data = self.preprocessing_es()
        jhs_processing_data = self.preprocessing_jhs()

        training_edited = self.merge_es_info(trainingdata, es_processing_data)
        training_edited = self.merge_jhs_info(trainingdata, jhs_processing_data)

        column_to_move = training_edited['單價']
        training_edited = training_edited.drop('單價', axis=1)
        training_edited['單價'] = column_to_move

        print(training_edited.head())

        training_edited.to_csv('training_data_edited.csv', index=False)

if __name__ == "__main__":
    edu_preproc = PreprocessingEdu()
    edu_preproc.create_edu_feature()
