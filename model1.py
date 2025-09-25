import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3)
        self.conv5 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv6 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(8, 10, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(10 * 3 * 3, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = x.view(-1, 10 * 3 * 3)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def build_model(device: torch.device):
    return Net().to(device)



