import torch
import torch.nn as nn
import torch.nn.functional as F


class HAR(nn.Module):
    def __init__(self):
        super(HAR, self).__init__()
        self.shared_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.shared_conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.shared_conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=2)
        self.shared_conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=2)

        self.private_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.private_conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.private_conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=2)
        self.private_conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=2)

        self.private_clf1 = nn.Linear(1184*2, 256)
        self.private_clf2 = nn.Linear(256, 6)

    def forward(self, x):
        g_x = F.relu(self.shared_conv1(x), inplace=True)
        g_x = F.max_pool1d(g_x, 2)
        g_x = F.relu(self.shared_conv2(g_x), inplace=True)
        g_x = F.max_pool1d(g_x, 2)
        g_x = F.relu(self.shared_conv3(g_x), inplace=True)
        g_x = F.max_pool1d(g_x, 2)
        g_x = F.relu(self.shared_conv4(g_x), inplace=True)
        g_x = F.max_pool1d(g_x, 2)
        g_x = g_x.flatten(start_dim=1)

        l_x = F.relu(self.private_conv1(x), inplace=True)
        l_x = F.max_pool1d(l_x, 2)
        l_x = F.relu(self.private_conv2(l_x), inplace=True)
        l_x = F.max_pool1d(l_x, 2)
        l_x = F.relu(self.private_conv3(l_x), inplace=True)
        l_x = F.max_pool1d(l_x, 2)
        l_x = F.relu(self.private_conv4(l_x), inplace=True)
        l_x = F.max_pool1d(l_x, 2)
        l_x = l_x.flatten(start_dim=1)

        feature = torch.cat((g_x, l_x), dim=-1)
        out = self.private_clf1(feature)
        out = self.private_clf2(out)
        return out


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