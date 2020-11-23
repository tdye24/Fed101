import argparse

DATASETS = ['femnist']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                        help='name of dataset',
                        choices=DATASETS,
                        required=True)

    parser.add_argument('-model',
                        help='name of model',
                        type=str,
                        required=True)

    parser.add_argument('-model-path',
                        help='path for storing the optimal global model',
                        type=str,
                        required=True)

    parser.add_argument('--num-rounds',
                        help='number of communication rounds',
                        type=int,
                        default=1000)

    parser.add_argument('--eval-interval',
                        help='communication rounds between two evaluation',
                        type=int,
                        default=20)

    parser.add_argument('--clients-per-round',
                        help='number of selected clients per round',
                        type=int,
                        default=1)

    parser.add_argument('--epoch',
                        help='epoch nums when clients train on data',
                        type=int,
                        default=1)

    parser.add_argument('--batch-size',
                        help='batch size when clients train on data',
                        type=int,
                        default=1)

    parser.add_argument('--mini-batch',
                        help='mini-batch when clients train on data',
                        type=float,
                        default=-1)

    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting',
                        type=int,
                        default=123)

    parser.add_argument('--lr',
                        help='learning rate for local optimizers',
                        type=float,
                        default=3e-4)

    return parser.parse_args()
