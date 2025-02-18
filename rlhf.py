import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
import random
from collections import deque
import time

from plotter import Plotter
from policy_network import PolicyNetwork
from reward_estimator import RewardEstimator

class RLHF:
    """
    Implements the a reinforcement learning with simulated human feedback algorithm.
    """
    def __init__(self, 
                 env: gym.Env = gym.make("CartPole-v1"),
                 real_reward: bool = False,
                 epsilon: float = -100,
                 policy_interval: int = 30,
                 reward_interval: int = 10,
                 reward_estimator_amount: int = 3,
                 policy_lr: float = 0.001,
                 reward_lr: float = 0.0001,
                 policy_clip_grad_norm_max: float = 1,
                 reward_clip_grad_norm_max: float = 1,
                 total_episodes: int = 600,
                 max_steps: int = 510,
                 discount_factor: float = 0.99,
                 memroy_size: int = 300,
                 exploration_start: float = 1.0,
                 exploration_decay: float = 0.95,
                 exploration_min: float = 0.01,
                 reward_writer = None,
                 label_writer = None,
                 plotting = False):
        """
        Args:
            env (gym.Env, optional): The gymnasium environment. Needs to discrete actions. Defaults to gym.make("CartPole-v1").
            real_reward (bool, optional): Set to True if the real reward should be used instead of an estimated reward. Defaults to False.
            epsilon (float, optional): The epsilon variable for the three-valued feedback function. Should it be less than or equal to -100, 
                then the two-valued feedback function is directly used instead. Can be ignored if real_reward is set to True. Defaults to -100.
            policy_interval (int, optional): The number of episodes the policy will be optimized successively before the reward estimator is 
                optimized. Should be at least a four, because the reward opimizer needs at least that many trajectories. The smaller this is set, 
                the lower should be the reward learning rate. Defaults to 30.
            reward_interval (int, optional): The number of simulated human feedbacks are collected before the policy is optimized again. 
                Defaults to 10.
            reward_estimator_amount (int, optional): The number of reward estimator networks to use, the estimated reward is averaged over these. 
                Can be ignored if real reward is set to True. Defaults to 3.
            policy_lr (float, optional): The learning rate for the policy optimizer. Defaults to 0.001.
            reward_lr (float, optional): The learning rate for the reward optimizer. 
                Can be ignored if real reward is set to True. Defaults to 0.0001.
            policy_clip_grad_norm_max (float, optional): The maximum gradient norm for the policy optimizer's gradient clipping. Defaults to 1.
            reward_clip_grad_norm_max (float, optional): The maximum gradient norm for the reward optimizers' gradient clipping. 
                Can be ignored if real reward is set to True. Defaults to 20.
            total_episodes (int, optional): The total number of episodes. Defaults to 600.
            max_steps (int, optional): The maximum number of steps per episode. If less than 1, then it is set to the maximum integer32. Defaults to 510.
            discount_factor (float, optional): The discount factor for the discounted reward calculation. Defaults to 0.99.
            memroy_size (int, optional): The trajectory deque's memory size. Can be ignored if real reward is set to True. Defaults to 250.
            exploration_start (float, optional): The initial exploration rate. Defaults to 1.0.
            exploration_decay (float, optional): The exploration rate decay. Defaults to 0.95.
            exploration_min (float, optional): The minimum exploration rate. Defaults to 0.01.
            reward_writer (RewardWriter, optional): A reward writer instance, which is used to write the reward onto a file. Defaults to None.
            label_writer (LabelWriter, optional): A label writer instance, which is used to write the feedback label occurrences onto a file. 
                Can be ignored if real reward is set to True. Defaults to None.
            plotting (bool, optional): If set to True, visualization of the training process performance will be enabled. Defaults to False.
        """
       
        self._USING_REAL_REWARD = real_reward

        #to ignore uncertainty region set eps to -100 or lower
        self._UNCERTAINTY_REGION_EPSILON = epsilon 
       
        self._POLICY_ITERATIONS_INTERVAL = policy_interval
        self._REWARD_ITERATIONS_INTERVAL = reward_interval

        self._CLIP_GRAD_NORM_MAX_NORM = policy_clip_grad_norm_max

        if total_episodes < policy_interval:
            raise ValueError(f"The total episodes ({total_episodes}) must be greater than the policy interval ({policy_interval})")

        self._TOTAL_EPISODES = total_episodes
        
        # if max steps is less than or 0, then set it to max int
        if max_steps <= 0:
            max_steps = int(torch.finfo(torch.int32).max)
        self._MAX_STEPS = max_steps
        
        self._DISCOUNT_FACTOR = discount_factor

        # Hyperparameters for exploration
        self._exploration_epsilon = exploration_start
        self._EXPLORATION_DECAY = exploration_decay
        self._EXPLORATION_MIN = exploration_min
        
        self._PLOTTING = plotting
        
        self._reward_writer = reward_writer
        self._label_writer = label_writer

        # Get the state dimension and number of possible discrete actions for this gymnasium environment
        self._env = env
        n_observations = self._env.observation_space.shape[0]
        n_actions = self._env.action_space.n

        # Initialize the policy network and its optimizer
        self._policy = PolicyNetwork(n_observations, n_actions)
        #self._policy_optimizer = optim.SGD(self._policy.parameters(), lr=policy_lr)
        self._policy_optimizer = optim.Adam(self._policy.parameters(), lr=policy_lr)

        # Create the trajectory memory
        self._trajectory_memory = deque(maxlen=memroy_size)

        if reward_estimator_amount < 1:
            raise ValueError("The reward estimator amount must be bigger than 0!")

        # Initialize the reward estimator, for its input dimension, we add the action dimension (1) to the evinronment's state dimension
        self._reward_estimator = RewardEstimator(n_observations + 1, self._trajectory_memory, amount=reward_estimator_amount, epsilon=epsilon, learning_rate=reward_lr, clip_grad_norm_max=reward_clip_grad_norm_max)

        # Initialize the plotter for visualization
        if self._PLOTTING:
            self._plt = Plotter(title="RLHF")

    def _select_action(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        """Uses the policy to select an action based on the state.

        Args:
            state (np.ndarray): The state of the environment.

        Returns:
            tuple[int, torch.Tensor]: The action returned from the policy. And a tensor of the log probabilities of all possible actions.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action, action_probs = self._policy(state_tensor)
        return action, action_probs

    def _collect_trajectory(self) -> tuple[list, torch.Tensor]:
        """Collects a trajectory and returns it along with the log probabilities of the actions taken.
        
        Returns:
            tuple[list, torch.Tensor]: The trajectory as a list of steps, with each step being a tuple of state, action, and reward. 
            And a tensor of the log probabilities of the actions taken in the trajectory.
        """
        state, _ = self._env.reset()
        trajectory = []
        log_probs = []
        
        # Perform steps in the environment
        for step in range(self._MAX_STEPS):
            # Select an action with the policy and also get the log probabilities of the possible actions
            action, probs = self._select_action(state)
            
            # Determine whether to select a random action
            if random.random() < self._exploration_epsilon:
                action = self._env.action_space.sample()
            
            # Perform the step with the selected action
            next_state, reward, terminated, truncated, _ = self._env.step(action)
            
            # Save the step
            trajectory.append((state, action, reward))
            
            # Save the log probability of the action taken
            log_prob = probs[action]
            log_probs.append(log_prob)
            
            # End the trajectory if the environment is done
            if terminated or truncated:
                break
            
            # Set the next state
            state = next_state      
        
        # Add the total reward to the reward writer if it is available
        if self._reward_writer is not None:
            self._reward_writer.add(sum(step[2] for step in trajectory))
        
        # Plot the total reward of the trajectory if plotting is enabled
        if self._PLOTTING:
            self._plt.plot_value(sum(step[2] for step in trajectory))
        
        # Update exploration epsilion to reduce the exploration over time
        self._exploration_epsilon = max(self._EXPLORATION_MIN, self._exploration_epsilon * self._EXPLORATION_DECAY)

        return trajectory, torch.stack(log_probs)

    def _calculate_policy_loss(self, trajectory: list, log_probs: torch.Tensor) -> torch.Tensor:
        """Calculates the policy loss using the REINFORCE algorithm by calculating the discounted reward

        Args:
            trajectory (list): A list of steps, with each step being a tuple of state, action, and reward.
            log_probs (torch.Tensor): The log probabilities of the actions taken in the trajectory.

        Returns:
            torch.Tensor: The policy loss.
        """
        with torch.no_grad():
            states, actions, real_rewards = zip(*trajectory)
            states = torch.tensor(np.array(states), dtype=torch.float32)
    
            # Estimate the reward with the reward estimator
            if not self._USING_REAL_REWARD:
                    rewards = self._reward_estimator.estimate_reward(states, actions)
            
                    # Calculate mse between real reward and estimated reward for plotting
                    if self._PLOTTING:
                        real_rewards = torch.tensor(np.array(real_rewards), dtype=torch.float32)
                        mse = torch.mean((rewards - real_rewards) ** 2)
                        self._plt.plot_value(mse.item(), 4, yLabel='Reward MSE')
                    
            # Use the real reward instead if the class argument real_reard was set True
            else:
                rewards = torch.tensor(np.array(real_rewards), dtype=torch.float32)
        
            # Calculate the discounted reward
            discounted_rewards = torch.zeros_like(rewards)
            R = 0
            for i in reversed(range(len(rewards))):
                R = rewards[i] + self._DISCOUNT_FACTOR * R
                discounted_rewards[i] = R
        
            # Normalize the discounted reward
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
                
        # Calculate the policy loss by summing the product of the log probabilities and the discounted rewards
        policy_loss = torch.sum(log_probs * discounted_rewards)
        
        # Turn the gradient ascend into a gradient descent by returning a negated the loss.
        return - policy_loss
    
    def _optimize_policy(self, iteration_amount: int = 1) -> None:
        """Collects and saves trajectories, then optimizes the policy with the REINFORCE algorithm by calculating the discounted reward 
        and backpropagating the policy loss. Also plots the policy loss if plotting is enabled.

        Args:
            iteration_amount (int, optional): The number of iteration steps. For each iteration step a trajectory will be collected and the policy will be optimized. Defaults to 1.
        """
        for _ in range(iteration_amount):
            # Collect a trajectory and save it to memory   
            trajectory, log_probs = self._collect_trajectory()
            self._trajectory_memory.append(trajectory)
            
            # Calculate the policy loss
            policy_loss = self._calculate_policy_loss(trajectory, log_probs)
            
            # Optimize the policy
            self._policy_optimizer.zero_grad()       
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._CLIP_GRAD_NORM_MAX_NORM)
            self._policy_optimizer.step()
            
            # Plot the policy loss in the subplot if enabled
            if self._PLOTTING:
                self._plt.plot_value(policy_loss.item(), 2, yLabel='Policy Loss')
        
    def _optimize_reward(self, query_amount: int = 1, epsiode: int = 0) -> None:
        """Collects simulated human feedbacks and optimizes the reward estimator. If plotting is enabled, will update the average reward 
        estmator loss on the subplot.

        Args:
            query_amount (int, optional): The amount of simulated human feedbacks to collect. Defaults to 1.
            epsiode (int, optional): The current episode number, used for writing to the label writer. Defaults to 0.
        """
        loss, labels_counter = self._reward_estimator.optimize_reward(query_amount)
                    
        if self._label_writer is not None:
            self._label_writer.add(epsiode, labels_counter)
                    
        if self._PLOTTING:
            self._plt.plot_value(loss, 3, average_over=5, xLabel=f"Episode / {self._POLICY_ITERATIONS_INTERVAL}", yLabel="Avg Estimator Loss")
            #self.plt.plot_values(labels_counter[2], 4, yLabel='Uncertainty Labels')

    def run(self, printing: bool = True) -> None:
        """Run the training process of this class. Trajectories and the policy will be optimzed. Feedback will be simulated and the reward 
        estimator will be optimized. These two processes will run alternately, as defined with the class arguments policy_interval and 
        reward_interval. Finishes once the total number of episodes is reached.
        
        Args:
            printing (bool, optional): If True, a message will be printed once the training is finished. Defaults to True.
        """
        start_time = time.time()
        
        # Training Loop
        for i in range(self._TOTAL_EPISODES // self._POLICY_ITERATIONS_INTERVAL):
            
            self._optimize_policy(self._POLICY_ITERATIONS_INTERVAL)
            
            # Optimize the reward estimator. If the real reward is used, this will be skipped.
            if not self._USING_REAL_REWARD:
                episode = (i+1) * self._POLICY_ITERATIONS_INTERVAL
                self._optimize_reward(self._REWARD_ITERATIONS_INTERVAL, episode)

        self._env.close()

        elapsed_time = time.time() - start_time
        
        if printing: 
            print(f"RLHF finished in {int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds!")
            
        if self._PLOTTING:
            self._plt.show()
            
