import torch
import torch.nn as nn
import torch.optim as optim

class GatingNetwork(nn.Module):
    def __init__(self, input_size=3):
        """
        Gating network that learns to weight expert predictions
        Args:
            input_size: Number of expert predictions to combine (default 3 for CNN, MLP, RNN)
        """
        super(GatingNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, input_size),
            nn.Softmax(dim=1)  # Ensures weights sum to 1
        )

    def forward(self, expert_predictions):
        """
        Combines predictions from multiple experts using learned weights
        Args:
            expert_predictions: Tensor of shape (batch_size, num_experts) containing predictions
                              from each expert model
        Returns:
            final_prediction: Weighted combination of expert predictions
            expert_weights: The weights assigned to each expert
        """
        # Get weights for each expert
        expert_weights = self.network(expert_predictions)
        
        # Weighted sum of expert predictions
        final_prediction = torch.sum(expert_predictions * expert_weights, dim=1, keepdim=True)
        
        return final_prediction, expert_weights


def combine_predictions(gating_network, cnn_pred, mlp_pred, rnn_pred):
    """
    Combine predictions from all experts using the trained gating network
    Args:
        gating_network: Trained gating network model
        cnn_pred: Predictions from CNN model
        mlp_pred: Predictions from MLP model
        rnn_pred: Predictions from RNN model
    Returns:
        final_predictions: Combined predictions
        weights: Weights assigned to each expert
    """
    # Stack predictions from all experts
    expert_predictions = torch.stack([cnn_pred, mlp_pred, rnn_pred], dim=1)
    
    # Get combined prediction and expert weights
    with torch.no_grad():
        final_predictions, weights = gating_network(expert_predictions)
    
    return final_predictions, weights
