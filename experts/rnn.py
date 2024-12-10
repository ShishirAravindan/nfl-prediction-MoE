import torch
import torch.nn as nn
import torch.optim as optim 


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(RNNModel, self).__init__()

        # Define the RNN layer (or LSTM)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Define the fully connected layer to map from hidden state to the output
        self.fc = nn.Linear(hidden_size, output_size)

        # Define a sigmoid activation for binary classification (pass/rush)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Get the RNN outputs
        # x shape: (batch_size, timesteps, num_features)
        out, (hn, cn) = self.rnn(x)  # out shape: (batch_size, timesteps, hidden_size)

        # We only care about the last timestep output (for classification)
        last_out = out[:, -1, :]  # (batch_size, hidden_size)

        # Pass through the fully connected layer and then sigmoid
        out = self.fc(last_out)  # Shape: (batch_size, output_size)
        out = self.sigmoid(out)  # Shape: (batch_size, 1)

        return out