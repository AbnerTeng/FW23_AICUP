"""
pipeline before modeling
"""
import os
from argparse import ArgumentParser
from .preproc import PreProc
from .features import (
    add_social_economic_feature,
    merge_lib_can_del,
    preprocessing_edu_v2,
    mean_dist,
    n_facilities_v2
)

def parse_args() -> ArgumentParser:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preproc = PreProc(
        f"{os.getcwd()}/data/training_data.csv"
    )
    preproc.main()
    preproced_data = preproc.data
        