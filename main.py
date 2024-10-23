import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = "cpu"

# The probabily function from the atari paper
def probability2(a, b):
    p = np.exp(a) / (np.exp(a) + np.exp(b))
    return (p, 1-p)

# My probabily function based on the atari paper
def probability3(a, b, eps):
    diff = np.abs(a-b)
    p_undecided = np.exp(2*eps) / (np.exp(diff) + np.exp(2*eps))
    q_undecided = 1 - p_undecided
    p_a, p_b = probability2(a, b)
    return (q_undecided * p_a, q_undecided * p_b, p_undecided)

# The deep neural network which will be used for the policy and reward estimator
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, softmax = True):
        super(DQN, self).__init__()
        self.softmax = softmax
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        if self.softmax:
            return F.softmax(self.layer3(x), dim=-1)
        else:
            return self.layer3(x)
        
# Create the environment
env = gym.make("CartPole-v1")
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# Initialize the policy network and its optimizer
policy = DQN(n_observations, n_actions) # takes in a state and returns the probabilities for actions
policy_optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Initialize the reward estimator network and its optimizer
reward_estimator = DQN(n_observations + 1, 1, softmax=False) # takes in a state and an action and returns an estimated reward
reward_optimizer = optim.Adam(reward_estimator.parameters(), lr=0.01)

# Function to collect a trajectory
def collect_trajectory(policy, max_steps=0):
    steps = 0
    state, _ = env.reset()
    trajectory = []
    done = False
    
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_probs = policy(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated or (steps > max_steps and max_steps > 0)
        trajectory.append((state, action, reward))
        state = next_state
    
    return trajectory

# Define the reward of the trajectory
def trajectory_reward(trajectory):
    # currently using the real reward not the estimated reward
    return sum(step[2] for step in trajectory)

# Simulated human feedback function
def simulated_human_feedback(trajectory1, trajectory2, eps=1):
    total_reward1 = trajectory_reward(trajectory1)
    total_reward2 = trajectory_reward(trajectory2)
    
    # Get the probabilities of the human feedback
    preferred_probs = probability3(total_reward1, total_reward2, eps)
    
    # Choose one feedback based on the probabilities
    preferred = torch.multinomial(torch.tensor(preferred_probs), 1).item()   
    
    # Convert the index to a real value
    if preferred == 2:
        preferred = 0.5
    return preferred, preferred_probs

# Optimze the policy
def optimize_policy(num_iterations):
    for _ in range(num_iterations):
        policy_optimizer.zero_grad()
        
        trajectory = collect_trajectory(policy)
        
        # Calculate discounted cumulative rewards
        gamma = 0.99  # Discount factor
        discounted_rewards = []
        cum_reward = 0
        for t in reversed(trajectory):
            reward = t[2] # using the real reward instead of an estimated reward
            # reward = reward_estimator(torch.cat((t[0], t[1]), 0)).item()
            cum_reward = reward + gamma * cum_reward
            discounted_rewards.insert(0, cum_reward)
        
        # Convert states and actions to tensors
        states = torch.tensor(np.array([step[0] for step in trajectory]), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array([step[1] for step in trajectory]), dtype=torch.int64, device=device)
                
        # Calculate action probabilities
        action_probs = policy(states)
        
        # Calculate loss
        loss = -(torch.log(action_probs.gather(1, actions.unsqueeze(1))) * torch.tensor(discounted_rewards, dtype=torch.float32)).sum()
        
        # Update policy        
        loss.backward()
        policy_optimizer.step()

# Optimze the estimated reward with the simulated human feedback
def optimize_reward(num_iterations):
    reward_optimizer.zero_grad()
    
    # Create multiple pairs of trajectories and the simulated human feedback on them
    feedbacks = []
    for iteration in range(num_iterations):
        trajectory1 = collect_trajectory(policy)
        trajectory2 = collect_trajectory(policy)
        feedback, p = simulated_human_feedback(trajectory1, trajectory2)
        feedbacks.append((feedback, p))
    
    # Calculate loss based on the atari paper formula in 2.2.3
    loss = 0
    for feedback, p in feedbacks:
        m1, m2 = 0, 0
        if feedback == 0:
            m1 = 1
        elif feedback == 1:
            m2 = 1
        elif feedback == 0.5:
            m1 = 0.5
            m2 = 0.5
                    
        p_tensor = torch.tensor(p[:2], dtype=torch.float32, requires_grad=True)
        loss -= m1 * torch.log(p_tensor[0]) + m2 * torch.log(p_tensor[1])
    
    # Backpropagate and update
    loss.backward()
    reward_optimizer.step()
    
# Test the trained policy and compare actual reward with the reward_estimator
def test():
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    mse = 0
    while not done:
        state_tensor = torch.FloatTensor(state)
        action_probs = policy(state_tensor)
        action = torch.argmax(action_probs).item()
        state, reward, terminated, truncated, _ = env.step(action)
        estimated_reward = reward_estimator(torch.cat((state_tensor, torch.tensor([action])), 0)).item()
        mse += (reward-estimated_reward) * (reward-estimated_reward)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    mse /= steps

    print(f"Total reward with trained policy in {steps} steps: {total_reward}")
    print(f"MSE for the estimated reward: {mse}")
    
# Training Loop
num_iterations = 500
for iteration in range(num_iterations):
    optimize_policy(30)
    optimize_reward(15)
    if (iteration + 1) % 20 == 0:
        print(f"\n{iteration + 1}/{num_iterations}:")
        test()

env.close()
print("finished!")