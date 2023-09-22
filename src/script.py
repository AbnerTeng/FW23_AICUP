import os
import pandas as pd
import plotly.express as px
from .utils import load_data
"""
data features:
1. No missing values
2. One useless column(備註)
3. Lots of categorical columns(8) (OHE / Label Encoding)
"""

path = f'{os.getcwd()}/data/training_data.csv'
data = load_data(path)
data = data.drop(['備註'], axis = 1)
print(data.info())
## print(data.isna().sum())
data['樓層比例'] = data['移轉層次'] / data['總樓層數']
for idx, name in enumerate(data):
    print(f'{name}: {len(data[name].unique())}')
"""
ID: 11751
縣市: 18
鄉鎮市區: 123
路名: 3059
土地面積: 9894
使用分區: 6
移轉層次: 37
總樓層數: 44
主要用途: 12
主要建材: 6
建物型態: 4
屋齡: 676
建物面積: 4664
車位面積: 1760
車位個數: 4
橫坐標: 10491
縱坐標: 10012
主建物面積: 6675
陽台面積: 2132
附屬建物面積: 2268
單價: 531
"""
nomial_dat = data.select_dtypes(include = ['float64', 'int64'])
cat_dat = data.select_dtypes(include = ['object'])
print(f'{nomial_dat.shape}, {cat_dat.shape}')

for col in nomial_dat.columns:
    fig = px.histogram(nomial_dat, x = col)
    ## fig.show()

## 幹我很想去猜他怎麼做去識別化的...
## 18 個縣市配上 123 個鄉鎮市區，不能直接拿這些全部去做 encoding，這樣會變超稀疏矩陣
## 路名有 3059 個，也不能直接拿去做 encoding


