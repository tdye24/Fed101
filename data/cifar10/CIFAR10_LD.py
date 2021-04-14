import json
import os
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

HOME = '/home/tdye/Fed101/data/cifar10'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_data = datasets.CIFAR10(root=HOME, train=True, transform=transform, download=True)
test_data = datasets.CIFAR10(root=HOME, train=False, transform=transform, download=True)
print(len(train_data))
print(len(test_data))


class CIFAR10_DATASET(Dataset):
    def __init__(self, use, ids):
        self._use = use
        self._ids = ids

    def __getitem__(self, item):
        if self._use == 'train':
            _x, _y = train_data[self._ids[item]]
        else:
            _x, _y = test_data[self._ids[item]]
        assert _x.shape == (3, 32, 32)
        _x = _x.float()
        _y = torch.tensor(_y).long()
        return _x, _y

    def __len__(self):
        return len(self._ids)


def get_cifar10_dataloaders(batch_size=10):
    with open(os.path.join(HOME, 'latent_distribution.json')) as f:
        client_ids = json.load(f)

    trainloaders = {}
    testloaders = {}

    for client_id, ids in client_ids.items():
        train_ids = ids['train']
        test_ids = ids['test']

        trainset = CIFAR10_DATASET(use='train', ids=train_ids)
        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testset = CIFAR10_DATASET(use='test', ids=test_ids)
        test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)
        trainloaders[client_id] = train_loader
        testloaders[client_id] = test_loader

    all_clients = list(client_ids.keys())
    return all_clients, trainloaders, testloaders


if __name__ == '__main__':
    clients, _trainloaders, _testloaders = get_cifar10_dataloaders()
    for client in clients:
        print("client", client)
        ls = []
        for _, (data, labels) in enumerate(_trainloaders[client]):
            ls.extend(list(np.array(torch.unique(labels))))
        print("train", np.unique(np.array(ls)))
        ls = []
        for _, (data, labels) in enumerate(_testloaders[client]):
            ls.extend(list(np.array(torch.unique(labels))))
        print("test", np.unique(np.array(ls)))
        print("=========")
