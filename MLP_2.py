import torch
import torch.nn as nn

class MLP_2(nn.Module):
    def __init__(self, input_size=3072, hidden_sizes=[512, 256, 128], output_size=10, dropout_rate=0.3):
        super(MLP_2, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_sizes[2], output_size)
        )

    def forward(self, x):
        return self.model(x)
