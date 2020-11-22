import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.model_utils import read_data, batch_data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(4 * 4 * 64, 2048)
        self.fc2 = nn.Linear(2048, 62)

    def forward(self, x):
        # 50x784 -> 50x1x28x28
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, 1, 28, 28))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_dir = os.path.join('..', '..', 'data', 'femnist', 'data', 'train')
    test_data_dir = os.path.join('..', '..', 'data', 'femnist', 'data', 'test')
    users, train_data_all, test_data_all = read_data(train_data_dir, test_data_dir)
    print("User Number: {}".format(len(users)))
    ui = 0
    for uid in users:
        # 每个客户端一个新模型
        net = Net().to(device)
        train_data, test_data = train_data_all[uid], test_data_all[uid]
        optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        BEST_ACC = 0
        for i in range(100):
            for batched_x, batched_y in batch_data(train_data, 50, 123):
                input_data, target_data = torch.tensor(batched_x).to(device), torch.tensor(batched_y).to(device)
                optimizer.zero_grad()
                output = net(input_data)
                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()

                net.eval()
                test_data_x = torch.tensor(test_data['x']).to(device)
                test_data_y = torch.tensor(test_data['y']).to(device)
                output = net(test_data_x)
                output = torch.argmax(output, dim=-1)
                acc = torch.sum(output == test_data_y).float() / len(test_data_y)
                if acc > BEST_ACC:
                    BEST_ACC = acc
        print("{} -> user: {}, training samples: {}, testing samples: {}, last acc: {}, best acc: {}".format(ui, uid, len(train_data['y']), len(test_data['y']), acc, BEST_ACC))
        ui += 1
