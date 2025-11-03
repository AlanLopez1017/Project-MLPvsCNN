import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size = 3072, hidden_sizes = [512, 256, 128], output_size = 10, dropout_rate=0.3):
        ''' Base configuration of the CNN model '''
        super(MLP, self).__init__()

        # Flatten layer
        self.flatter = nn.Flatten()

        # Hidden layers
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second hidden layer
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)

        # Third hidden layer
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)


    def forward(self, x):

        # (batch_size, 3, 32, 32) -> (batch_size, 3072)
        x = self.flatter(x)

        # First hidden layer, ReLU and dropout
        x = self.dropout1(F.relu(self.fc1(x)))

        # Second hidden layer, ReLU and dropout
        x = self.dropout2(F.relu(self.fc2(x)))

        # Third hidden layer, ReLU and dropout
        x = self.dropout3(F.relu(self.fc3(x)))

        # Output layer
        x = self.fc4(x)

        return x


