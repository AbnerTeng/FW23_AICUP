"""
Main execution file
"""
import os
from argparse import ArgumentParser
from .preproc import PreProc
from .modeling import Modeling


def argument_parser() -> ArgumentParser:
    """
    Argument parser
    """
    parser = ArgumentParser()
    parser.add_argument(
        '--dims', type=int, default=10,
        help='Number of dimensions (features)'
    )
    parser.add_argument(
        '--method', type=str, default='RFR',
        help="Type of ensemble method"
    )
    parser.add_argument(
        '--type', type=str, default='train',
        help="Type of execution"
    )
    parser.add_argument(
        '--save', type=bool, default=False,
        help='Save predict output or not'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()
    preproc = PreProc(
        f"{os.getcwd()}/data/external_data/",
        f"{os.getcwd()}/data/training_data.csv",
        args, dims=0
    )
    ## preproc.save_files() -> save the preprocessed files
    train_session = Modeling(
        f"{os.getcwd()}/data/training_data.csv",
        f"{os.getcwd()}/data/testing_data.csv"
    )
    xtr, ytr, xvl, yvl = train_session.split_data(args.ratio)
    if args.type == "train":
        train_session.train(args.method, xtr, ytr, xvl, yvl)
    if args.type == "fine_tune":
        train_session.fine_tune(args.method, xtr, ytr, xvl, yvl)
    