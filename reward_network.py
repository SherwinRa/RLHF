import torch
import torch.nn as nn
    
class RewardNetwork(nn.Module):
    """A neural network for the reward estimation for a reinforcement learning environment.

    The reward network takes in a tensor of a state and action and outputs an estimated reward based on them.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """Initialization of the reward network

        Args:
            input_dim (int): The dimension of the environment's state and action summed together
            hidden_dim (int, optional): The number of neurons in the hidden layers. Defaults to 64
        """
        super(RewardNetwork, self).__init__()   
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            #nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through this reward network. Returns an estimated reward.
        
        Args:
            x (torch.Tensor): The values of a state and an action in one tensor.
        
        Returns:
            reward (torch.Tensor): An estimated reward for the given state and action.
        """
        reward = self.fc(x)
        self._check_for_nan(reward)
        return reward
    
    def _check_for_nan(self, values: torch.Tensor) -> None:
        """Checks for nan values and raises a RuntimeError if they are present.
        
        Args:
            values (torch.Tensor): Values to check for nan.
        """
        if torch.isnan(values).any():
            raise RuntimeError("A Reward Estimator network returned nan values, this could be because of a gradient explosion. Consider lowering the reward learning rate.")
