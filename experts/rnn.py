import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class OffensiveCoordinatorRNN(nn.Module):
    def __init__(self, feature_size=7, num_players=11, sequence_size=10,
                 hidden_size=128, num_layers=2):
        super(OffensiveCoordinatorRNN, self).__init__()

        self.num_players = num_players
        self.feature_size = feature_size
        self.sequence_size = sequence_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN for temporal patterns (LSTM)
        self.rnn = nn.LSTM(
            input_size=feature_size * num_players,  # Combine all player features
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )

        # Output layer for binary classification
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, X):
        """
        Forward pass for input X.
        X shape: (batch_size, feature_size, num_players, sequence_size)
        """
        batch_size, num_players, sequence_size, feature_size = X.shape

        # Reshape input to (batch_size, sequence_size, feature_size * num_players)
        X = X.permute(0, 3, 2, 1).reshape(batch_size, sequence_size, -1)
        # RNN
        rnn_output, _ = self.rnn(X)  # Shape: (batch_size, sequence_size, hidden_size)

        # Use the last hidden state for classification
        last_hidden_state = rnn_output[:, -1, :]  # Shape: (batch_size, hidden_size)

        logits = self.output_layer(last_hidden_state)  # Shape: (batch_size, 1)

        # Apply sigmoid for binary classification
        output = torch.sigmoid(logits)  # Shape: (batch_size, 1)

        return output
    
    def train(self, train_loader, val_loader=None, num_epochs=100, learning_rate=0.001):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_acc = 0.0
            total_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.float()
                y_batch = y_batch.float().view(-1, 1)
                optimizer.zero_grad()

                # Forward pass: Get predictions through the LSTM
                output = self(X_batch)
                loss = criterion(output, y_batch) # Backpropagation and optimization
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                # Step the optimizer
                optimizer.step()
                acc = accuracy(y_batch, output)

                running_loss += loss.item()
                running_acc += acc
                total_batches += 1

            train_losses.append(running_loss / total_batches)
            train_accuracies.append(running_acc / total_batches)

            # Print the average loss and accuracy for this epoch
            avg_loss = running_loss / total_batches
            avg_acc = running_acc / total_batches
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}")

            # Validation metrics, if provided
            if val_loader is not None:
                val_loss = 0.0
                val_acc = 0.0
                total_val_batches = 0
                
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch = X_val_batch.float()
                        y_val_batch = y_val_batch.float().view(-1, 1)
                        
                        val_outputs = self(X_val_batch)
                        batch_val_loss = criterion(val_outputs, y_val_batch)
                        batch_val_acc = accuracy(y_val_batch, val_outputs)
                        
                        val_loss += batch_val_loss.item()
                        val_acc += batch_val_acc
                        total_val_batches += 1
                
                avg_val_loss = val_loss / total_val_batches
                avg_val_acc = val_acc / total_val_batches
                val_losses.append(avg_val_loss)
                val_accuracies.append(avg_val_acc)
                print(f"\tValidation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc:.4f}")

        return {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies, 
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }

def accuracy(y_true, y_pred):
    y_pred = torch.round(torch.sigmoid(y_pred))
    correct = (y_pred == y_true).float()
    acc = correct.sum() / len(correct)
    return acc.item()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_prime(x):
    return 1 - tanh(x) ** 2

class man_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(man_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate parameters
        self.W_i = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U_i = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # Forget gate parameters
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U_f = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        # Cell gate parameters
        self.W_c = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U_c = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        # Output gate parameters
        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U_o = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, hidden_state=None, cell_state=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if cell_state is None:
            cell_state = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # Get the t-th input vector

            # Input gate
            i_t = torch.sigmoid(x_t @ self.W_i.T + hidden_state @ self.U_i.T + self.b_i)
            # Forget gate
            f_t = torch.sigmoid(x_t @ self.W_f.T + hidden_state @ self.U_f.T + self.b_f)
            # Output gate
            o_t = torch.sigmoid(x_t @ self.W_o.T + hidden_state @ self.U_o.T + self.b_o)
            # Candidate cell state
            g_t = torch.tanh(x_t @ self.W_c.T + hidden_state @ self.U_c.T + self.b_c)

            # Update cell state and hidden state
            cell_state = f_t * cell_state + i_t * g_t
            hidden_state = o_t * torch.tanh(cell_state)

            outputs.append(hidden_state.unsqueeze(1))  # Append output for this time step

        outputs = torch.cat(outputs, dim=1)  # Concatenate along time dimension
        return outputs, (hidden_state, cell_state)