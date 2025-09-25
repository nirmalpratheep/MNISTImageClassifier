import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 8, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 8, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(8)

        self.conv5 = nn.Conv2d(8, 16, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(16, 12, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(12)

        self.conv7 = nn.Conv2d(12, 10, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1) # WRONG 0.05-0.1

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


def build_model(device: torch.device):
    return Net().to(device)



