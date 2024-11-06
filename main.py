import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import sys

from plotter import Plotter
from dqn_models import DQN

device = "cpu"
min_positive_float32 = torch.finfo(torch.float32).tiny

EPISODES = 50
POLICY_ITERATIONS_PER_EPISODE = 20
REWARD_ITERATIONS_PER_EPISODE = 10

POLICY_LEARNING_RATE = 0.01 # 0.001
REWARD_LEARNING_RATE = 0.00001
LEARNING_RATE = 0.001
TRAJECTORY_MEMORY_SIZE = 100
BATCH_SIZE = 15

# Hyperparameters for exploration
GAMMA = 0.99 # Discount factor
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Create the environment
env = gym.make("CartPole-v1")
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# Initialize the policy network and its optimizer
policy = DQN(n_observations, n_actions) # takes in a state and returns the probabilities for actions
policy_optimizer = optim.Adam(policy.parameters(), lr=POLICY_LEARNING_RATE)

# Initialize the reward estimator network and its optimizer
reward_estimator = DQN(n_observations + 1, 1) # takes in a state and an action and returns an estimated reward
reward_optimizer = optim.Adam(reward_estimator.parameters(), lr=REWARD_LEARNING_RATE)

#policy_criterion = nn.SmoothL1Loss()
policy_criterion = nn.MSELoss()

# Create the trajectory replay memory
trajectory_memory = deque(maxlen=TRAJECTORY_MEMORY_SIZE)

# Init Plotter for visualization of the reward
plt = Plotter()

# Function to select an action
def select_action(state, policy):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_probs = policy(state_tensor)
        #action_probs = F.softmax(action_probs, dim=-1)
        #action = torch.multinomial(action_probs, 1).item()
        action = action_probs.argmax().item()
        return action

# Function to collect a trajectory
def collect_trajectory(policy, max_steps=0):
    global EPSILON
    steps = 0
    state, _ = env.reset()
    trajectory = []
    done = False
    
    while not done:
        # Select a random action or use the policy to select an action
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = select_action(state, policy)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated or (steps > max_steps and max_steps > 0)
        trajectory.append((state, action, reward))
        state = next_state
    
    # Plot the total reward, in the cartpole env the total steps equals the total reward
    plt.plot_durations(steps) #tmp
    
    # Reduce epsilion to reduce the exploration over time
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    return trajectory

# Optimze the policy
def optimize_policy(num_iterations):
    for _ in range(num_iterations):
        # Collect a trajectory and save it to memory   
        trajectory = collect_trajectory(policy)
        trajectory_memory.append(trajectory)
        
        # If there are enough trajectories saved, optimize the policy with a sample of them
        if len(trajectory_memory) > BATCH_SIZE:
            batch = random.sample(trajectory_memory, BATCH_SIZE)
        
            # Prepare batch data
            states = []
            next_states = []
            actions = []
            #rewards = []
            dones = []
            for traj in batch:
                states.extend([step[0] for step in traj[:-1]])
                next_states.extend([step[0] for step in traj[1:]])
                actions.extend([step[1] for step in traj[:-1]])
                #rewards.extend([step[2] for step in traj[:-1]])
                dones.extend(0 for step in traj[:-2])
                dones.append(1)
            
            # Convert the data to tensors
            states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
            actions = torch.tensor(np.array(actions), dtype=torch.long, device=device)
            #rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device) # uses real reward, but should be using the estimated reward
            dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device)
            
            # Calculate an estimate of the reward with the reward estimator
            state_action_pairs = torch.cat([states, actions.unsqueeze(1).float()], dim=1)
            rewards = reward_estimator(state_action_pairs).squeeze()
        
            # Calculate state-action values
            action_probs = policy(states)
            state_action_values = action_probs.gather(1, actions.unsqueeze(1))
                     
            # Calculate expected state-action values
            next_state_action_values = policy(next_states).max(1)[0]
            expected_state_action_values = rewards + GAMMA * next_state_action_values * (1 - dones)
        
            # Calculate loss      
            loss = policy_criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            #plt.plot_durations(loss.item()) #tmp

            # Update policy 
            policy_optimizer.zero_grad()       
            loss.backward()
            policy_optimizer.step()

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

# My probabily function based on the atari paper with torch
def probability3torch(estimated_reward1, estimated_reward2, eps=1.0):
    diff = torch.abs(estimated_reward1 - estimated_reward2)
    eps = torch.tensor(eps)
    p_undecided = torch.exp(2*eps) / (torch.exp(diff) + torch.exp(2*eps))
    q_undecided = 1 - p_undecided
    p_a = torch.exp(estimated_reward1) / (torch.exp(estimated_reward1) + torch.exp(estimated_reward2))
    p_b = 1 - p_a
    return torch.stack([q_undecided * p_a, q_undecided * p_b, p_undecided])

# Estimate an reward based on a state and action with the reward estimator
def estimate_reward(state, action):    
    state_tensor = torch.FloatTensor(state)
    estimated_reward = reward_estimator(torch.cat((state_tensor, torch.tensor([action])), 0)).item()
    return estimated_reward

# Define the reward of a trajectory
def trajectory_reward(trajectory):
    return sum(step[2] for step in trajectory)

# Define the estimated reward of a trajectory
def trajectory_estimated_reward(trajectory):
    estimated_reward = sum(reward_estimator(torch.cat((torch.FloatTensor(step[0]), torch.tensor([step[1]])), 0)) for step in trajectory)
    
    # Check the reward estimator for gradient explosion
    if torch.isnan(estimated_reward):
        for name, param in reward_estimator.named_parameters():
            if 'weight' in name:
                print(f"{name}:")
                print(param.data)
                print()
        print(f"{estimated_reward=}")
        sys.exit()
    return estimated_reward

# Simulated human feedback function
def simulated_human_feedback(trajectory1, trajectory2, eps=1, estimate=False):
    if estimate:
        total_reward1 = trajectory_estimated_reward(trajectory1)
        total_reward2 = trajectory_estimated_reward(trajectory2)
    else:
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
    
# Optimze the estimated reward with the simulated human feedback
def optimize_reward(num_iterations):
    if len(trajectory_memory) <= 2:
        return

    loss = 0
    for iteration in range(num_iterations):
        trajectory1, trajectory2 = random.sample(trajectory_memory, 2)
        
        # Calculate estimated rewards for both trajectories
        estimated_reward1 = trajectory_estimated_reward(trajectory1)
        estimated_reward2 = trajectory_estimated_reward(trajectory2)
                
        # Get simulated human feedback (using actual rewards)
        feedback, _ = simulated_human_feedback(trajectory1, trajectory2, estimate=False)
        
        # Calculate probabilities based on estimated rewards
        probs = probability3torch(estimated_reward1, estimated_reward2)
        
        # Ensure no zero probabilities
        probs = torch.clamp(probs, min=min_positive_float32)

        # Calculate loss based on the atari paper formula in 2.2.3
        if feedback == 0:
            loss -= torch.log(probs[0])
        elif feedback == 1:
            loss -= torch.log(probs[1])
        else:  # feedback == 0.5
            loss -= 0.5 * (torch.log(probs[0]) + torch.log(probs[1]))
        
    # Backpropagate and update
    reward_optimizer.zero_grad()
    loss.backward()
    reward_optimizer.step()
    
    #plt.plot_durations(loss.item()) #tmp

# Training Loop
for episode in range(EPISODES):
    optimize_policy(POLICY_ITERATIONS_PER_EPISODE)
    optimize_reward(REWARD_ITERATIONS_PER_EPISODE)

env.close()
print("finished!")
plt.show()
