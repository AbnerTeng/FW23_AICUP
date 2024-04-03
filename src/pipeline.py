"""
pipeline before modeling
"""
import os
import json
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from .preproc import PreProc
from .model.stacking import stacking
from .model.tuning import ParamTuner


def parse_args() -> ArgumentParser:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of nearest neighbors"
    )
    parser.add_argument(
        "--radius", type=int, default=500,
        help="Radius for facilities"
    )
    parser.add_argument(
        "--tune", action="store_true"
    )
    parser.add_argument(
        "--model_to_tune", type=str, default="xgb"
    )
    return parser.parse_args()


if __name__ == "__main__":
    with open(f"{os.getcwd()}/columns.json", encoding="utf-8") as json_file:
        cols = json.load(json_file)

    TARGET_PATH = f"{os.getcwd()}/data/training_data.csv"
    args = parse_args()
    train_preproc = PreProc(
        f"{os.getcwd()}/data/training_data.csv",
        f"{os.getcwd()}/data/train_feat.csv",
        f"{os.getcwd()}/data/train_output.csv",
        "train"
    )
    private_preproc = PreProc(
        f"{os.getcwd()}/data/private_dataset_org.csv",
        f"{os.getcwd()}/data/private_dataset.csv",
        None,
        "private"
    )
    test_preproc = PreProc(
        f"{os.getcwd()}/data/public_dataset.csv",
        f"{os.getcwd()}/data/test_feat.csv",
        None,
        "test"
    )
    train_x, train_y = train_preproc.select_features(**cols)
    private_x = private_preproc.select_features(**cols)
    test_x = test_preproc.select_features(**cols)
    x_tr, x_vl, y_tr, y_vl, test_x, private_x = train_preproc.encode_cat_features(
        cols['cat_cols'], train_x, test_x, private_x, train_y
    )
    stack_model = stacking()
    stack_model.fit(x_tr, y_tr)
    y_pred = stack_model.predict(x_vl)
    y_pred = np.exp(y_pred)
    mape = mean_absolute_percentage_error(y_vl, y_pred)
    print(f"MAPE: {mape} * 100")

    if args.tune:
        tuner = ParamTuner(x_tr, y_tr, x_vl, y_vl)
        model_dict = {
            "xgbr": tuner.opjective_xgb,
            "lgbmr": tuner.objective_lgbm,
            "catbr": tuner.objective_cat
        }
        best_params = tuner.optimize(model_dict[args.model_to_tune])

        if args.model_to_tune not in model_dict.keys():
            raise ValueError("Invalid model to tune")

        tuner.save_yml(f"{os.getcwd()}/configs/{args.model_to_tune}.yaml", best_params)