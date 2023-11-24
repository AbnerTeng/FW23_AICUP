"""
feature importance script

if dat_type == 'train':
    feature, output = feature_select(
        data, '單價', self.dims, self.model
    )
    return feature, output
else:
    return data



to merge the final output

You'll have the `y_pred` from predicting, it is a np.ndarray with shape (n, 1)
And you can create an empty dataframe:
    df = pd.DataFrame(
        'ID': test_id,
        'predicted_price': y_pred
    )
"""
import os
import pandas as pd
prv_data = pd.read_csv(f"{os.getcwd()}/data/private_dataset.csv")
prv_data['縣市_雲林縣'] = 0
print(prv_data.columns)
prv_data.to_csv(f"{os.getcwd()}/data/private_dataset.csv", index=False)