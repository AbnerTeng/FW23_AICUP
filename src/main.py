"""
Main execution file
"""
from argparse import ArgumentParser
from .predict import Predict


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
        '--save', type=bool, default=False,
        help='Save predict output or not'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()
    predict = Predict(args.dims)
    x_train, x_test, y_train, y_test = predict.train_test_split()
    print(f"Size of every sets: \
            {x_train.shape},\
            {x_test.shape},\
            {y_train.shape},\
            {y_test.shape}"
    )
    pred, mae = predict.train()
    print(mae)
    if args.save:
        pred.to_csv('data/pred.csv', index=False)
    