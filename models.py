import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2) # -> 32x28x28
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2) # -> 32x14x14
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2) # -> 64x14x14
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) # -> 64x7x7

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10) # 10 classes for Fashion-MNIST

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
