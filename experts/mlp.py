import torch.nn as nn
import torch.optim as optim
import torch
from etl.transform_mlp import load_mlp_features


class GameContextMLP(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_classes=1, pretrained=False):
        super(GameContextMLP, self).__init__()
        
        # Three hidden layer MLP
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )

        if pretrained:
            self.load_pretrained()

    def forward(self, x):
        return self.layers(x)

    def load_pretrained(self):
        checkpoint = torch.load('../checkpoints/game_context.pth')
        self.load_state_dict(checkpoint)
        # self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def train_model(self, train_loader, val_loader=None, epochs=10, learning_rate=0.001):
        """
        Train the MLP model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            correct_preds = 0
            total_preds = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_X)

                # Ensure target has correct shape
                batch_y = batch_y.float().view(-1, 1) 
                loss = criterion(outputs, batch_y)
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                correct_preds += (predicted == batch_y).sum().item()
                total_preds += batch_y.size(0)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct_preds / total_preds * 100
            
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

            # Validation
            if val_loader is not None:
                self.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self(batch_X)
                        # Ensure target has correct shape
                        batch_y = batch_y.float().view(-1, 1)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        # Calculate validation accuracy
                        val_predicted = (outputs > 0.5).float()
                        val_correct += (val_predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total * 100
                
                val_losses.append(avg_val_loss)
                val_accuracies.append(val_accuracy)
                
                print(f'\t Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        return {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }