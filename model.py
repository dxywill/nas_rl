# generic model design

import torch.nn as nn
import torch.nn.functional as F


class NASModel(nn.Module):
    def __init__(self, actions):
        super(NASModel, self).__init__()
        # unpack the actions from the list
        self.kernel_1, self.filters_1, self.kernel_2, self.filters_2 = actions.tolist()
        # input size 3 * 32 * 32, use default stride=1 and padding=0
        # w and h could be calculated using the below equation
        # w = (w - filter_size + 2*p)/stride + 1 = w - filter_size + 1
        # thus if use 2*2 max pooling, (w - filter_size + 1) % 2 = 0
        # filter size could be in [3, 5, 7], also, we limit filter numbers to be [8, 16, 32]
        self.conv1 = nn.Conv2d(3, self.filters_1, self.kernel_1)
        # input filters_1 * (33 - kernel_1) * (33 - kernel_1)
        self.pool = nn.MaxPool2d(2, 2)
        # input filters_1 * ((33 - kernel_1) / 2) * ((33 - kernel_1) / 2)
        self.conv2 = nn.Conv2d(self.filters_1, self.filters_2, self.kernel_2)
        # input filters_2 * ((33 - kernel_1) / 2 - kernel_2 + 1) * ((33 - kernel_1) / 2 - kernel_2 + 1)


        self.tmp = int(self.filters_2 * ((33 - self.kernel_1) / 2 - self.kernel_2 + 1) *
                             ((33 - self.kernel_1) / 2 - self.kernel_2 + 1))
        self.fc1 = nn.Linear(self.tmp, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.tmp)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
