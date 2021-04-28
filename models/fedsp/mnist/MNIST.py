import torch
import torch.nn as nn


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.shared_global_feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2)
        )

        self.private_local_feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2)
        )

        self.private_fc = torch.nn.Sequential(
            torch.nn.Linear(64*7*7*2, 512),  # 乘2因为global_feat和local_feat拼在一起
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        global_feat = self.shared_global_feature(x)
        local_feat = self.private_local_feature(x)

        global_feat = global_feat.flatten(start_dim=1)
        local_feat = local_feat.flatten(start_dim=1)

        feature = torch.cat((global_feat, local_feat), dim=-1)
        output = self.private_fc(feature)
        return output


if __name__ == '__main__':
    model = MNIST()
    _x = torch.rand((50, 1, 28, 28))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('shared'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))