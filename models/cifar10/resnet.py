import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(4 * 4 * 64, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # 50x784 -> 50x1x28x28
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, 1, 28, 28))
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x

