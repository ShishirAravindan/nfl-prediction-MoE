import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FormationExpertCNN(nn.Module):
    def __init__(self, input_channels=4, input_height=158, input_width=300, pretrained=False):
        super(FormationExpertCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Dynamically calculate flattened size
        self.flatten_size = self._get_flatten_size(input_channels, input_height, input_width)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        if pretrained:
            self.load_pretrained()

    def _get_flatten_size(self, channels, height, width):
        # Simulate forward pass to calculate the size
        x = torch.zeros(1, channels, height, width)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        return x.numel()

    def forward(self, x):
        # Convolutional and pooling layers
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        # Flatten the tensor
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Sigmoid activation for binary classification
        x = torch.sigmoid(x)
        return x
    
    def load_pretrained(self):
        checkpoint = torch.load(f'../checkpoints/CNNmodel.pth')
        self.load_state_dict(checkpoint)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def train(self, X_train, y_train, X_val=None, y_val=None, num_epochs=50, 
              batch_size=32, alpha=0.001):
        
        # Convert numpy arrays to torch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
    
        # Create dataloader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=alpha)
        
        # Training metrics
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            epoch_loss = 0
            correct_preds = 0
            total_preds = 0

            for _, (images, labels) in enumerate(train_loader):
                # forward pass
                images = images.permute(0, 3, 1, 2)
                outputs = self(images)
                loss = criterion(outputs, labels.unsqueeze(1))
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                correct_preds += (predicted == labels.unsqueeze(1)).sum().item()
                total_preds += labels.size(0)
                
                # backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_accuracy = correct_preds / total_preds * 100
            
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
            
            # Validation metrics, if provided
            if X_val is not None and y_val is not None:
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
                    val_outputs = self.forward(X_val_tensor)
                    val_loss = criterion(val_outputs, torch.FloatTensor(y_val).unsqueeze(1))
                    
                    # Calculate validation accuracy
                    val_predicted = (val_outputs > 0.5).float()
                    val_correct = (val_predicted == torch.FloatTensor(y_val).unsqueeze(1)).sum().item()
                    val_accuracy = val_correct / len(y_val) * 100
                    
                    val_losses.append(val_loss.item())
                    val_accuracies.append(val_accuracy)
                    
                    print(f'Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        return {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies, 
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }