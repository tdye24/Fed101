import torch
import torch.nn as nn
import torch.nn.functional as F


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
    net = Net()
    order_dict_1 = net.state_dict()['conv1.weight']
    order_dict_2 = net.state_dict()['conv1.weight']
    order_dict_bias = net.state_dict()['conv1.bias']
    print(order_dict_1)
    print(order_dict_bias)
    xx = net.state_dict()
    xx['conv1.weight'] = order_dict_1 + order_dict_2
    print(xx)
    net.load_state_dict(xx)