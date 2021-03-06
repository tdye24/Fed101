import torch
import torch.nn as nn
import torch.nn.functional as F


class FLICKR(nn.Module):
    def __init__(self):
        super(FLICKR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(5 * 5 * 16, 60)
        self.drop_out1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(60, 32)
        self.drop_out2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):
        # 100x3x32x32
        x = F.relu(self.conv1(x), inplace=True)
        # 100x32x28x28
        x = F.max_pool2d(x, 2, 2)
        # 100x32x14x14
        x = F.relu(self.conv2(x), inplace=True)
        # 100x64x10x10
        x = F.max_pool2d(x, 2, 2)
        # 100x64x5x5
        x = x.view(-1, 5 * 5 * 16)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.drop_out1(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.drop_out2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = FLICKR()
    x = torch.rand((50, 3, 32, 32))
    output = model(x)
    print(f'{x.shape}->{output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))
