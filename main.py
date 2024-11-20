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

TOTAL_EPISODES = 500 # 1000
POLICY_ITERATIONS_PER_BATCH = 50
REWARD_ITERATIONS_PER_BATCH = 20
UNCERTAINTY_REGION_EPSILON = 0.5 # 0.5
IGNORE_UNCERTAINTY_REGION = False

POLICY_LEARNING_RATE = 0.001 # 0.001
REWARD_LEARNING_RATE = 0.0002 # 0.0001
TRAJECTORY_MEMORY_SIZE = 100
#BATCH_SIZE = 15

# Hyperparameters for exploration
GAMMA = 0.99 # Discount factor
EPSILON = 1.0
EPSILON_DECAY = 0.995 # 0.995
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

# Create the trajectory replay memory
trajectory_memory = deque(maxlen=TRAJECTORY_MEMORY_SIZE)

# Init Plotter for visualization of the reward
plt = Plotter(ignore_eps=IGNORE_UNCERTAINTY_REGION, eps=UNCERTAINTY_REGION_EPSILON, policy_lr=POLICY_LEARNING_RATE, reward_lr=REWARD_LEARNING_RATE, policy_reward_ratio=f"{POLICY_ITERATIONS_PER_BATCH}/{REWARD_ITERATIONS_PER_BATCH}")

# The probabily function from the atari paper
def probability2(a, b):
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    
    p = 1 / (1.0 + torch.exp(b-a))
    return torch.stack([p, 1-p])

# My probabily function based on the atari paper
def probability3(a, b, eps=0.5, ignore_eps=IGNORE_UNCERTAINTY_REGION):
    if ignore_eps:
        return probability2(a, b)
    
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    eps = torch.as_tensor(eps)

    diff = torch.abs(a - b)
    pq_undecided = probability2(2*eps, diff)
    p_undecided, q_undecided = pq_undecided[0], pq_undecided[1]
    p_a, p_b = probability2(a, b) * q_undecided

    return torch.stack([p_a, p_b, p_undecided])

# Estimate an reward based on a tensor of states and a tensor of actions with the reward estimator
def estimate_reward(states, actions):
    states = torch.as_tensor(states)
    actions = torch.as_tensor(actions)
    state_action_pairs = torch.cat([states, actions.unsqueeze(1).float()], dim=1)
    rewards = reward_estimator(state_action_pairs).squeeze()
    return rewards

# Define the reward of a trajectory
def trajectory_reward(trajectory):
    return sum(step[2] for step in trajectory)

# Define the estimated reward of a trajectory
def trajectory_estimated_reward(trajectory):
    states, actions, rewards = zip(*trajectory)
    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    estimated_reward = torch.sum(estimate_reward(states, actions))
    #estimated_reward = sum(reward_estimator(torch.cat((torch.FloatTensor(step[0]), torch.tensor([step[1]])), 0)) for step in trajectory)
    
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
def simulated_human_feedback(trajectory1, trajectory2, eps=0.5):
    total_reward1 = trajectory_reward(trajectory1)
    total_reward2 = trajectory_reward(trajectory2)
        
    # Get the probabilities of the human feedback
    preferred_probs = probability3(total_reward1, total_reward2, eps)
        
    # Choose one feedback based on the probabilities
    #preferred = torch.multinomial(torch.tensor(preferred_probs), 1).item()
    preferred = torch.multinomial(preferred_probs, 1).item()
    
    # Convert the index to a real value
    if preferred == 2:
        preferred = 0.5
    return preferred, preferred_probs


# Function to select an action
def select_action(state, policy):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    action_probs = policy(state_tensor)
    #action_probs = torch.softmax(action_probs, dim=-1)
    #action = torch.multinomial(action_probs, 1).item()
    action_probs = torch.log_softmax(action_probs, dim=-1)
    action = torch.distributions.Categorical(logits=action_probs).sample().item()
    #action = action_probs.argmax().item()
    return action, action_probs

# Function to collect a trajectory
def collect_trajectory(policy, max_steps=0):
    global EPSILON
    state, _ = env.reset()
    trajectory = []
    log_probs = []
    done = False
    
    for step in range(max_steps):
        action, probs = select_action(state, policy)
        
        # Select a random action or use the policy to select an action
        if random.random() < EPSILON:
            action = env.action_space.sample()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory.append((state, action, reward))
        
        #log_prob = torch.log(probs[action].clamp(min=min_positive_float32))
        log_prob = probs[action]
        log_probs.append(log_prob)
        
        if terminated or truncated:
            break
        
        state = next_state      
    
    # Plot the total reward, in the cartpole env the total steps equals the total reward
    plt.plot_durations(step + 1) #tmp
    #plt.plot_durations(trajectory_reward(trajectory)) #tmp
    
    # Reduce epsilion to reduce the exploration over time
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    return trajectory, torch.stack(log_probs)

# Optimize the policy with gradient decent
def optimize_policy(num_iterations):
    for _ in range(num_iterations):
        # Collect a trajectory and save it to memory   
        trajectory, log_probs = collect_trajectory(policy, max_steps=510)
        trajectory_memory.append(trajectory)
        
        states, actions, _ = zip(*trajectory)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        #with torch.no_grad():
            #rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
        
        #with torch.no_grad():
        rewards = estimate_reward(states, actions)
        
        discounted_rewards = torch.zeros_like(rewards)
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + GAMMA * R
            discounted_rewards[i] = R
        
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
                
        policy_loss = -torch.sum(log_probs * discounted_rewards)

        policy_optimizer.zero_grad()       
        policy_loss.backward()
        policy_optimizer.step()
        
        #plt.plot_durations(policy_loss.item())
    
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
        with torch.no_grad():
            feedback, _ = simulated_human_feedback(trajectory1, trajectory2, eps=UNCERTAINTY_REGION_EPSILON)
        
        # Calculate probabilities based on estimated rewards
        probs = probability3(estimated_reward1, estimated_reward2, eps=UNCERTAINTY_REGION_EPSILON)
        
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

# Compare the estimated reward with the actual reward by calculating the MSE
def compare_estimate_reward(num_iterations):
    estimated_probs_list = []
    real_probs_list = []
    for iteration in range(num_iterations):
        with torch.no_grad():
            trajectory1, trajectory2 = random.sample(trajectory_memory, 2)
            estimated_reward1 = trajectory_estimated_reward(trajectory1)
            estimated_reward2 = trajectory_estimated_reward(trajectory2)
            real_reward1 = trajectory_reward(trajectory1)
            real_reward2 = trajectory_reward(trajectory2)
            estimated_probs = probability3(estimated_reward1, estimated_reward2, eps=UNCERTAINTY_REGION_EPSILON)
            real_probs = probability3(real_reward1, real_reward2, eps=UNCERTAINTY_REGION_EPSILON)
            estimated_probs_list.append(estimated_probs.cpu().numpy())
            real_probs_list.append(real_probs.cpu().numpy())
    
    mse = np.mean((np.array(estimated_probs_list) - np.array(real_probs_list))**2)
    print(f"MSE: {mse}")
    #plt.plot_durations(mse)
        


# Training Loop
for batch_index in range(TOTAL_EPISODES // POLICY_ITERATIONS_PER_BATCH):
    #episode = batch_index * POLICY_ITERATIONS_PER_BATCH
    optimize_policy(POLICY_ITERATIONS_PER_BATCH)
    optimize_reward(REWARD_ITERATIONS_PER_BATCH)
    #compare_estimate_reward(15)

env.close()
print("finished!")
plt.show()
