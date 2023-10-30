"""
Main execution function
"""
from argparse import ArgumentParser
from .predict import Predict

def arguments() -> ArgumentParser:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        '--dims', type=int, default=10,
        help="Number of features to be selected"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
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
    # pred.to_csv('data/pred.csv', index  = False)