import json
import os
import sys
from collections import defaultdict

import numpy as np

# models
from models.fedavg.cifar10.CIFAR10 import CIFAR10 as FedAVG_CIFAR10
from models.fedavg.femnist.FEMNIST import FEMNIST as FedAVG_FEMNIST
from models.fedavg.mnist.MNIST import MNIST as FedAVG_MNIST


from models.fedsp.cifar10.CIFAR10 import CIFAR10 as FedSP_CIFAR10
from models.fedsp.femnist.FEMNIST import FEMNIST as FedSP_FEMNIST
from models.fedsp.mnist.MNIST import MNIST as FedSP_MNIST


class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def batch_data(data, batch_size, seed):
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])

    # clients = list(sorted(data.keys()))
    return clients, data


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_clients, train_data = read_dir(train_data_dir)
    test_clients, test_data = read_dir(test_data_dir)
    # 可能clients读入的顺序不一样
    assert train_clients.sort() == test_clients.sort()

    return train_clients, train_data, test_data


def load_model(algorithm='fedavg', model_name=''):
    model = None
    if algorithm == 'fedsp':
        if model_name == 'femnist':
            model = FedSP_FEMNIST()
        elif model_name == 'cifar10':
            model = FedSP_CIFAR10()
        elif model_name == 'mnist':
            model = FedSP_MNIST()
        else:
            print("Unimplemented Model!")
            exit(0)
    elif algorithm == 'fedavg':
        if model_name == 'femnist':
            model = FedAVG_FEMNIST()
        elif model_name == 'cifar10':
            model = FedAVG_CIFAR10()
        elif model_name == 'mnist':
            model = FedAVG_MNIST()
        else:
            print("Unimplemented Model!")
            exit(0)
    assert model is not None
    return model
