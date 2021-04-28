import torch
import torch.nn as nn
import torch.nn.functional as F


class HAR(nn.Module):
    def __init__(self):
        super(HAR, self).__init__()
        self.g_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.g_conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.g_conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=2)
        self.g_conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=2)

        self.l_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.l_conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.l_conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=2)
        self.l_conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=2)

        self.clf1 = nn.Linear(1184*2, 256)
        self.clf2 = nn.Linear(256, 6)

        self.critic1 = nn.Linear(1184, 256)
        self.critic2 = nn.Linear(256, 10)

    def forward(self, x):
        g_x = F.relu(self.g_conv1(x), inplace=True)
        g_x = F.max_pool1d(g_x, 2)
        g_x = F.relu(self.g_conv2(g_x), inplace=True)
        g_x = F.max_pool1d(g_x, 2)
        g_x = F.relu(self.g_conv3(g_x), inplace=True)
        g_x = F.max_pool1d(g_x, 2)
        g_x = F.relu(self.g_conv4(g_x), inplace=True)
        g_x = F.max_pool1d(g_x, 2)
        g_x = g_x.flatten(start_dim=1)

        l_x = F.relu(self.l_conv1(x), inplace=True)
        l_x = F.max_pool1d(l_x, 2)
        l_x = F.relu(self.l_conv2(l_x), inplace=True)
        l_x = F.max_pool1d(l_x, 2)
        l_x = F.relu(self.l_conv3(l_x), inplace=True)
        l_x = F.max_pool1d(l_x, 2)
        l_x = F.relu(self.l_conv4(l_x), inplace=True)
        l_x = F.max_pool1d(l_x, 2)
        l_x = l_x.flatten(start_dim=1)

        feature = torch.cat((g_x, l_x), dim=-1)
        out = self.clf1(feature)
        out = self.clf2(out)

        g_critic1_out = self.critic1(g_x)
        g_critic2_out = F.softmax(self.critic2(g_critic1_out), dim=1)

        l_critic1_out = self.critic1(l_x)
        l_critic2_out = F.softmax(self.critic2(l_critic1_out), dim=1)

        return g_critic2_out, l_critic2_out, out


if __name__ == '__main__':
    import torch
    model = HAR()
    _x = torch.rand((50, 1, 561))
    _g_critic2_out, _l_critic2_out, _output = model(_x)
    print(f'{_x.shape}->{_g_critic2_out.shape}')
    print(f'{_x.shape}->{_l_critic2_out.shape}')
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('g') or key.startswith('critic'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))