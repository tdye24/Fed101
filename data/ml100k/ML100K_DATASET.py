import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

BASE = '/home/tdye/Fed101/data/ml100k/data'
# TODO(改成绝对路径)
data_path = BASE + r'/data.csv'
labels_path = BASE + r'/labels.csv'

all_data = np.genfromtxt(data_path, delimiter=',', skip_header=True)
all_labels = np.genfromtxt(labels_path, delimiter=',')

# TODO 打乱数据集
train_data = all_data[0: 90000]
train_labels = all_labels[0: 90000]
test_data = all_data[90000:]
test_labels = all_labels[90000:]


class ML100KDATASET(Dataset):
    def __init__(self, data, labels, ids):
        self._data = data
        self._labels = labels
        self._ids = ids  # 该client包含数据的下标列表

    def __getitem__(self, item):
        _x = self._data[self._ids[item]]
        _y = self._labels[self._ids[item]]
        assert len(_x) == 22
        # assert len(_y) == 1

        _X = torch.tensor(_x).float()
        _y = torch.tensor(_y).long() - 1  # 评分是1-5，5分类，所有要所有标签减1
        return _x, _y

    def __len__(self):
        return len(self._ids)


def _get_ml100k_dataloaders(use='train', batch_size=10, num_clients=100):
    users = [i for i in range(100)]
    if use == 'train':
        data = train_data
        labels = train_labels
    else:
        data = test_data
        labels = test_labels
    # iid，划分到100个客户端
    ids = [i for i in range(len(data))]
    np.random.shuffle(ids)
    dataloaders = []
    for i in range(num_clients):
        samples_per_client = len(data) // num_clients
        client_ids = ids[samples_per_client * i: samples_per_client * (i + 1)]
        dataset = ML100KDATASET(data=data, labels=labels, ids=client_ids)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        dataloaders.append(data_loader)
    if use == 'train':
        assert len(dataloaders[0]) * batch_size == 900
    else:
        assert len(dataloaders[0]) * batch_size == 100
    return users, dataloaders


def get_ml100k_dataloaders(batch_size=10, num_clients=100):
    train_users, trainloaders = _get_ml100k_dataloaders(use='train', batch_size=batch_size, num_clients=num_clients)
    test_users, testloaders = _get_ml100k_dataloaders(use='test', batch_size=batch_size, num_clients=num_clients)
    assert train_users.sort() == test_users.sort()
    return train_users, trainloaders, testloaders


if __name__ == '__main__':
    _users, _trainloaders, _testloaders = get_ml100k_dataloaders(batch_size=10, num_clients=100)
