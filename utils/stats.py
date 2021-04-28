import argparse
import json
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys

from scipy import io
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

# data
# cifar10
from data.cifar10.CIFAR10_DATASET import get_cifar10_dataloaders
# cifar10_dirichlet
from data.cifar10.CIFAR10_DIRICHLET import load_partition_data_cifar10 as get_cifar10_dirichlet_dataloaders
# cifar10_latent_distribution
from data.cifar10.CIFAR10_LD import get_cifar10_dataloaders as get_cifar10_ld_dataloaders

# cifar100
from data.cifar100.CIFAR100_DATASET import get_cifar100_dataloaders

# mnist_malposition
from data.mnist.MNIST_DATASET_malposition import get_mnist_dataloaders as get_mnist_malposition_dataloaders
# mnist_wo_malposition
from data.mnist.MNIST_DATASET_wo_malposition import get_mnist_dataloaders as get_mnist_wo_malposition_dataloaders
# mnist_fedml
from data.mnist.MNIST_DATASET_fedml import get_mnist_dataloaders as get_mnist_fedml_dataloaders
# har
from data.har.HAR_DATASET import get_har_dataloaders

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                    help='name of dataset to parse; default: cifar10;',
                    type=str,
                    choices=['mnist_malposition', 'cifar10', 'cifar100', 'har'],
                    default='cifar10')

args = parser.parse_args()


def load_data(name):
    users = []
    train_num_samples = []
    test_num_samples = []
    trainloaders = None
    testloaders = None
    if name == 'cifar10':
        users, trainloaders, testloaders = get_cifar10_dataloaders()
    elif name == 'cifar100':
        users, trainloaders, testloaders = get_cifar100_dataloaders()
    elif name == 'mnist_malposition':
        users, trainloaders, testloaders = get_mnist_malposition_dataloaders()
    elif name == 'har':
        users, trainloaders, testloaders = get_har_dataloaders()

    for client in trainloaders.keys():
        trainloader = trainloaders[client]
        testloader = testloaders[client]
        train_num_samples.append(len(trainloader.sampler))
        test_num_samples.append(len(testloader.sampler))
    return users, train_num_samples, test_num_samples


def print_dataset_stats(name):
    users, train_num_samples, test_num_samples = load_data(name)
    num_users = len(users)

    print('####################################')
    print('DATASET: %s' % name)
    print('%d users' % num_users)

    print("training set")
    print('%d samples (total)' % np.sum(train_num_samples))
    print('%.2f samples per user (mean)' % np.mean(train_num_samples))
    print('num_samples (std): %.2f' % np.std(train_num_samples))
    print('num_samples (std/mean): %.2f' % (np.std(train_num_samples) / np.mean(train_num_samples)))
    print('num_samples (skewness): %.2f' % stats.skew(train_num_samples))

    print("testing set")
    print('%d samples (total)' % np.sum(test_num_samples))
    print('%.2f samples per user (mean)' % np.mean(test_num_samples))
    print('num_samples (std): %.2f' % np.std(test_num_samples))
    print('num_samples (std/mean): %.2f' % (np.std(test_num_samples) / np.mean(test_num_samples)))
    print('num_samples (skewness): %.2f' % stats.skew(test_num_samples))

    print("train & test set")
    train_and_test_num_samples = [train_num_samples[i] + test_num_samples[i] for i in range(len(train_num_samples))]
    print('%d samples (total)' % np.sum(train_and_test_num_samples))
    print('%.2f samples per user (mean)' % np.mean(train_and_test_num_samples))
    print('num_samples (std): %.2f' % np.std(train_and_test_num_samples))
    print('num_samples (std/mean): %.2f' % (np.std(train_and_test_num_samples) / np.mean(train_and_test_num_samples)))
    print('num_samples (skewness): %.2f' % stats.skew(train_and_test_num_samples))


print_dataset_stats(args.name)
