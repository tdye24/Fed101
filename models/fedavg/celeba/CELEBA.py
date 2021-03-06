import torch
import torch.nn as nn


class CELEBA(nn.Module):
    def __init__(self):
        super(CELEBA, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(32*7*7, 2)
        )

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out_flatten = cnn_out.flatten(start_dim=1)
        output = self.fc(cnn_out_flatten)
        return output


if __name__ == '__main__':
    model = CELEBA()
    _x = torch.rand((50, 3, 84, 84))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

