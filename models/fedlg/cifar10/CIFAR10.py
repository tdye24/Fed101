import torch.nn as nn
import torch.nn.functional as F


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.private_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.private_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.private_drop_out1 = nn.Dropout(p=0.25)

        self.shared_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.shared_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.shared_drop_out2 = nn.Dropout(p=0.25)

        self.shared_fc1 = nn.Linear(5 * 5 * 64, 512)
        self.shared_drop_out3 = nn.Dropout(p=0.5)
        self.shared_fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 50x3x32x32
        x = F.relu(self.private_conv1(x), inplace=True)
        x = F.relu(self.private_conv2(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.private_drop_out1(x)

        x = F.relu(self.shared_conv3(x), inplace=True)
        x = F.relu(self.shared_conv4(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.shared_drop_out2(x)

        x = x.view(-1, 5 * 5 * 64)
        x = F.relu(self.shared_fc1(x), inplace=True)
        x = self.shared_drop_out3(x)
        x = self.shared_fc2(x)
        return x


if __name__ == '__main__':
    import torch
    model = CIFAR10()
    _x = torch.rand((50, 3, 32, 32))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('shared'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))