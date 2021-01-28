import argparse

ALGORITHM = ['fedavg', 'fedprox']
DATASETS = ['femnist', 'cifar10', 'cifar100', 'synthetic_iid', 'mnist', 'mnist-prox', 'ml100k', 'femnist-fedml']
PARTITIONING = ['iid', 'non-iid']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-algorithm',
                        help='algorithm-[fedavg/fedprox]',
                        choices=ALGORITHM,
                        required=True)

    parser.add_argument('-dataset',
                        help='name of dataset',
                        choices=DATASETS,
                        required=True)

    parser.add_argument('-partitioning',
                        help='iid or non-iid',
                        choices=PARTITIONING,
                        required=True)

    parser.add_argument('-model',
                        help='name of model',
                        type=str,
                        required=True)

    parser.add_argument('--pretrain-model',
                        help='path of pre-trained model',
                        type=str,
                        default=None)

    parser.add_argument('--num-rounds',
                        help='number of communication rounds',
                        type=int,
                        default=1000)

    parser.add_argument('--eval-interval',
                        help='communication rounds between two evaluation',
                        type=int,
                        default=20)

    parser.add_argument('--all-clients-num',
                        help='number of all clients',
                        type=int,
                        default=100)

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

    parser.add_argument('--lr-decay',
                        help='decay rate for learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--decay-step',
                        help='decay rate for learning rate',
                        type=float,
                        default=200)
    return parser.parse_args()
