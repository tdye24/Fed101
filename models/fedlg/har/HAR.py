import torch.nn as nn
import torch.nn.functional as F


class HAR(nn.Module):
    def __init__(self):
        super(HAR, self).__init__()
        self.private_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.private_conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.shared_conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=2)
        self.shared_conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=2)
        self.shared_clf1 = nn.Linear(1184, 256)
        self.shared_clf2 = nn.Linear(256, 6)

    def forward(self, x):
        x = F.relu(self.private_conv1(x), inplace=True)
        x = F.max_pool1d(x, 2)
        x = F.relu(self.private_conv2(x), inplace=True)
        x = F.max_pool1d(x, 2)
        x = F.relu(self.shared_conv3(x), inplace=True)
        x = F.max_pool1d(x, 2)
        x = F.relu(self.shared_conv4(x), inplace=True)
        x = F.max_pool1d(x, 2)
        x = x.flatten(start_dim=1)
        x = self.shared_clf1(x)
        x = self.shared_clf2(x)
        return x


if __name__ == '__main__':
    import torch
    model = HAR()
    _x = torch.rand((50, 1, 561))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('shared'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))