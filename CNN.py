import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes = 10, dropout_rate = 0.3):
        super(CNN, self).__init__()

        # Convolutional block 1: 32 filters, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride = 2) # 32x32 -> 16x16

        # Convolutional block 2: 64 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride = 2) # 16x16 -> 8x8

        # Convolutional block 3: 128 filters, 3x3 kernel
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride = 2) # 8x8 -> 4x4

        # Convolutional block 4: 256 filters, 3x3 kernel
        #self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding = 1)
        #self.bn3 = nn.BatchNorm2d(256)
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride = 2) # 4x4 -> 2x2

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers

        # 4 pooling: 2x2x256 = 1024
        #self.fc1 = nn.Linear(256 * 2 * 2, 256)
        #self.dropout1 = nn.Dropout(dropout_rate)

        # 3 pooling: 4x4x128 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        # Steps: Conv -> BN -> ReLU -> Pool

        # Block 1: 
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2: 
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3: 
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten
        x = self.flatten(x)

        # Fully connected layer 
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # Output layer
        x = self.fc2(x)

        return x
