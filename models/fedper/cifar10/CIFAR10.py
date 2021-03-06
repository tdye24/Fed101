import torch
import torch.nn as nn
import torch.nn.functional as F


# class CIFAR10(nn.Module):
#     cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
#
#     def __init__(self):
#         super(CIFAR10, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.layers = self._make_layers(in_planes=32)
#         self.linear = nn.Linear(1024, 10)
#
#     def _make_layers(self, in_planes):
#         layers = []
#         for x in self.cfg:
#             out_planes = x if isinstance(x, int) else x[0]
#             stride = 1 if isinstance(x, int) else x[1]
#             layers.append(Block(in_planes, out_planes, stride))
#             in_planes = out_planes
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layers(out)
#         out = F.avg_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return F.log_softmax(out, dim=1)
#
#
# class Block(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1):
#         super(Block, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         return out

class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.personalization = nn.Sequential(
            nn.Linear(64*5*5, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        base_output = self.base(x)
        base_output = base_output.flatten(start_dim=1)
        output = self.personalization(base_output)
        return output


if __name__ == '__main__':
    model = CIFAR10()
    _x = torch.rand((50, 3, 32, 32))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))


