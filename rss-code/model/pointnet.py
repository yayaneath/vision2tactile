import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Caution: Since PointNet modules use batch normalization, batches
# are expected to have at least 2 samples

class STNkD(nn.Module):
    def __init__(self, input_channels=3):
        super(STNkD, self).__init__()

        self.k = input_channels

        self.conv1 = nn.Conv1d(self.k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.k * self.k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        eye_np = np.eye(self.k).flatten().astype(np.float32)
        iden = Variable(torch.from_numpy(eye_np))
        iden = iden.view(1, self.k * self.k).repeat(batch_size, 1)

        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, self.k, self.k)

        return x

# PointNet Global Feature calculator
class PointNetFeat(nn.Module):
    def __init__(self, input_channels=3):
        super(PointNetFeat, self).__init__()

        self.stn = STNkD(input_channels=input_channels)
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        transformed = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transformed)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x

class PointNet(nn.Module):
    def __init__(self, input_channels=3, output_vars=2):
        super(PointNet, self).__init__()

        self.feat = PointNetFeat(input_channels=input_channels)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_vars)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        #return F.log_softmax(x, dim=1)
        return x