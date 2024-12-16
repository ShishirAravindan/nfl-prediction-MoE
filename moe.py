import torch
import torch.nn as nn
import torch.optim as optim
from experts.cnn import FormationExpertCNN
from experts.rnn import OffensiveCoordinatorRNN
from experts.mlp import GameContextMLP
from experts.gating_network import GatingNetwork

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) model for NFL play prediction.
    
    Combines predictions from three pre-trained expert models:
    - Formation Expert (CNN): Analyzes spatial patterns in player formations
    - Offensive Coordinator (RNN): Processes temporal patterns in player movements
    - Game Context (MLP): Considers situational features
    
    The gating network learns to weight expert predictions based on their reliability
    for each input. Expert models are frozen during training; only the gating network
    is trained.
    
    Attributes:
        formation_expert (FormationExpertCNN): Pre-trained CNN for formation analysis
        offensive_coordinator (OffensiveCoordinatorRNN): Pre-trained RNN for temporal patterns
        game_context (GameContextMLP): Pre-trained MLP for game context
        gating_network (GatingNetwork): Trainable network for weighting expert predictions
    """
    def __init__(self):
        """
        Initialize MoE with pre-trained experts and gating network
        """
        super(MixtureOfExperts, self).__init__()
        
        # Initialize experts with pretrained=True
        self.formation_expert = FormationExpertCNN(pretrained=True)
        self.offensive_coordinator = OffensiveCoordinatorRNN(pretrained=True)
        self.game_context = GameContextMLP(pretrained=True)
        
        # Gating network is initialized without pretrained weights
        self.gating_network = GatingNetwork(3)

    def forward(self, cnn_features, mlp_features, rnn_features):
        """
        Forward pass through MoE
        Args:
            batch_data: Dict containing inputs for each expert
        Returns:
            final_pred: Final weighted prediction
            expert_weights: Weights assigned to each expert (for analysis)
            expert_preds: Dictionary of individual expert predictions (for analysis)
        """
        with torch.no_grad():  # No gradients needed for frozen experts
            # Get predictions from frozen experts
            formation_pred = self.formation_expert(cnn_features)
            coordinator_pred = self.offensive_coordinator(rnn_features)
            context_pred = self.game_context(mlp_features)
        
        # Stack predictions for gating network
        expert_preds = torch.stack([formation_pred, coordinator_pred, context_pred], dim=1).squeeze(-1)
        
        # Get final prediction and weights from gating network
        final_pred, expert_weights = self.gating_network(expert_preds)
        
        return final_pred, expert_weights, {
            'formation': formation_pred,
            'coordinator': coordinator_pred,
            'context': context_pred
        }

    def train(self, train_loader, num_epochs=100, learning_rate=0.001):
        """
        Train the gating network while keeping experts frozen
        Args:
            train_loader: DataLoader containing dict with 'cnn', 'rnn', 'mlp' inputs and labels
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        criterion = nn.BCELoss()
        # Only optimize gating network parameters
        optimizer = optim.Adam(self.gating_network.parameters(), lr=learning_rate)
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
            
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            epoch_weights = []
            expert_correct = {'cnn': 0, 'rnn': 0, 'mlp': 0}
            
            for batch_data, labels in train_loader:
                # Get features from batch
                cnn_features = batch_data['cnn'].permute(0, 3, 1, 2)
                mlp_features = batch_data['mlp']
                rnn_features = batch_data['rnn']

                # Forward pass
                final_pred, _, _ = self(cnn_features, mlp_features, rnn_features)
                
                # Compute loss
                loss = criterion(final_pred, labels)

                # Backward pass and optimize (only gating network)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                predicted = (final_pred > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            # Print epoch statistics
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')