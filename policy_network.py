import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """A neural network for the policy for a reinforcement learning environment with discrete actions.

    The policy network takes in a state and outputs a discrete action and the log probabilities of all possible actions.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initialization of the policy network

        Args:
            state_dim (int): The dimension of the environment's state
            action_dim (int): The number of discrete actions possible in the environment
            hidden_dim (int, optional): The number of neurons in the hidden layers. Defaults to 128
        """
        super(PolicyNetwork, self).__init__()   
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
                
    def forward(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Forward pass through this policy network. Returns one action and the log probabilities of all possible actions.
        
        Args:
            x (torch.Tensor): State of the environment.
        
        Returns:
            action (int): A discrete action of the environment.
            action_log_probs (torch.Tensor): The Log probabilities of all possible actions.
        """
        action_log_probs = self.fc(x)
        action_log_probs = torch.log_softmax(action_log_probs, dim=-1)
        
        self._check_for_nan(action_log_probs)
        
        action = torch.distributions.Categorical(logits=action_log_probs).sample().item()
        return action, action_log_probs
    
    def _check_for_nan(self, values: torch.Tensor) -> None:
        """Checks for nan values and raises a RuntimeError if they are present.
        
        Args:
            values (torch.Tensor): Values to check for nan.
        """
        if torch.isnan(values).any():
            raise RuntimeError("The policy network returned nan values, this could be because of a gradient explosion. Consider lowering the policy learning rate.")

        