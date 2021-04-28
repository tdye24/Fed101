import numpy as np
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))

from pyemd import emd
from data.cifar10.CIFAR10_DATASET import get_cifar10_dataloaders as get_cifar10_dataloaders
from data.cifar10.CIFAR10_DIRI import get_cifar10_dirichlet_dataloaders
from data.cifar10.CIFAR10_LD import get_cifar10_dataloaders as get_cifar10_ld_dataloaders

from data.mnist.MNIST_DATASET_wo_malposition import get_mnist_dataloaders
from data.mnist.MNIST_LD import get_mnist_dataloaders as get_mnist_ld_dataloaders
from data.mnist.MNIST_DATASET_fedml import get_mnist_dataloaders as get_mnist_power_law_dataloaders


parser = argparse.ArgumentParser()

parser.add_argument('--name',
                    help='name of dataset to evaluate; default: cifar10;',
                    type=str,
                    choices=['mnist', 'mnist_fedml', 'mnist_ld', 'cifar10', 'cifar10_dirichlet', 'cifar10_ld'],
                    default='mnist')

args = parser.parse_args()
dataset = args.name

N_class = None
users, trainloaders, testloaders = None, None, None

if dataset == 'mnist':
    users, trainloaders, testloaders = get_mnist_dataloaders(batch_size=10)
    N_class = 10
elif dataset == 'mnist_fedml':
    users, trainloaders, testloaders = get_mnist_power_law_dataloaders(batch_size=10)
    N_class = 10
elif dataset == 'mnist_ld':
    users, trainloaders, testloaders = get_mnist_ld_dataloaders(batch_size=10)
    N_class = 10

elif dataset == 'cifar10_ld':
    users, trainloaders, testloaders = get_cifar10_ld_dataloaders(batch_size=10)
    N_class = 10
elif dataset == 'cifar10':
    users, trainloaders, testloaders = get_cifar10_dataloaders(batch_size=10)
    N_class = 10
elif dataset == 'cifar10_dirichlet':
    users, trainloaders, testloaders = get_cifar10_dirichlet_dataloaders(users_num=30, alpha=0.3)
    N_class = 10
else:
    pass

total_dataset_distribution = {i: 0 for i in range(N_class)}
total_dataset_num_samples = 0
users_histogram = []
for user in users:

    train_loader = trainloaders[user]
    distribution = {i: 0 for i in range(N_class)}

    num_samples = len(train_loader.sampler)
    total_dataset_num_samples += num_samples

    for step, (data, labels) in enumerate(train_loader):
        for label in labels:
            distribution[label.item()] += 1

    histogram = []
    for label, num in distribution.items():
        total_dataset_distribution[label] += num
        histogram.append(num / num_samples)
    users_histogram.append((num_samples, histogram))

total_dataset_histogram = np.zeros(10)
for num, histogram in users_histogram:
    total_dataset_histogram = np.add(total_dataset_histogram, num*np.array(histogram))

total_dataset_histogram /= total_dataset_num_samples
print(total_dataset_histogram)

distances = []

for num, histogram in users_histogram:
    d = np.linalg.norm(total_dataset_histogram-histogram, ord=1)
    distances.append(d)

print("avg emd: ", sum(distances) / len(distances))
