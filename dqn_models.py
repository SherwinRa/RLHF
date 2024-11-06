import torch.nn as nn
#import torch.nn.functional as F

# The deep neural network which will be used for the policy and reward estimator
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)