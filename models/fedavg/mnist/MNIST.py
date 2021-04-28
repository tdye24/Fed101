import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.shared_cov1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.shared_cov2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.shared_fc1 = nn.Linear(64 * 7 * 7, 512)
        self.shared_fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.shared_cov1(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.shared_cov2(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = x.flatten(start_dim=1)
        x = F.relu(self.shared_fc1(x), inplace=True)
        x = self.shared_fc2(x)
        return x


if __name__ == '__main__':
    model = MNIST()
    x = torch.rand((50, 1, 28, 28))
    output = model(x)
    print(f'{x.shape}->{output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

