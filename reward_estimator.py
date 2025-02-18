import numpy as np
import torch
import torch.optim as optim
import random
from collections import deque

from reward_network import RewardNetwork
import feedback_functions as ff

class RewardEstimator():
    """
    This class handles the reward estimation for a reinforcement learning problem with simulated human feedback.
    """
    def __init__(self, input_dim: int, trajectory_memory, amount: int = 3, learning_rate: float = 0.00005, clip_grad_norm_max: float = 1, epsilon: float = -100):
        """
        Args:
            input_dim (int): the dimension of the input for the reward network, this should be the dimensions of the einvironment's state and action summed together.
            trajectory_memory (Sequence): A sequence of trajectories, with each trajectory being a list of steps, with each step being a tuple of state, action, and reward.
            amount (int, optional): The number of reward estimator networks to use. Defaults to 3.
            learning_rate (float, optional): The learning rate to use for the reward estimator networks. Defaults to 0.00005.
            clip_grad_norm_max (float, optional): The maximum gradient norm for the reward optimizers' gradient clipping. Defaults to 1.
            epsilon (float, optional): The epsilon variabe to use for the uncertainty region. This is used for the three valued feedback function. 
                If epsilon is less than or equal to -100, then the two-valued feedback function is directly used. Defaults to -100.
            
        Raises:
            ValueError: If the input dimension is less than 1.
            ValueError: If no trajectory memory is provided.
            ValueError: If the amount of reward estimators is less than 1.
        """
        if input_dim < 1:
            raise ValueError("Input dimension must be bigger than 0!")
        
        if trajectory_memory is None:
            raise ValueError("A trajectory memory must be provided!")
        
        if amount < 1:
            raise ValueError("Amount of reward estimators must be bigger than 0!")
        
        self._UNCERTAINTY_REGION_EPSILON = epsilon
        self._MIN_POSITIVE_FLOAT32 = torch.finfo(torch.float32).tiny
        self._CLIP_GRAD_NORM_MAX_NORM = clip_grad_norm_max
        self._FEEDBACK_MEMORY_SIZE = 5000
        
        self._trajectory_memory = trajectory_memory
        self._feedback_memory = deque(maxlen=self._FEEDBACK_MEMORY_SIZE)
        
        # Set up the reward estimators and their optimizers
        self._reward_estimators = [RewardNetwork(input_dim) for _ in range(amount)]
        self._reward_estimator_optimizers = [optim.Adam(reward_estimator.parameters(), lr=learning_rate) for reward_estimator in self._reward_estimators]
        
    def _estimate_reward_with_single_estimator(self, reward_estimator, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Estimates rewards based on a tensor of states and a tensor of actions. Only one reward estimators is used.

        Args:
            reward_estimator (RewardNetwork): The reward estimator network to use.
            states (torch.Tensor): A tensor of states.
            actions (torch.Tensor): A tensor of actions.

        Returns:
            torch.Tensor: A tensor of the estimated rewards from only one reward estimator.
        """
        states = torch.as_tensor(states)
        actions = torch.as_tensor(actions)
        state_action_pairs = torch.cat([states, actions.unsqueeze(1).float()], dim=1)
        rewards = reward_estimator(state_action_pairs).squeeze()
        return rewards
    
    def estimate_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Estimates rewards based on a tensor of states and a tensor of actions. All reward estimators are used and an averaged rewards are returned

        Args:
            states (torch.Tensor): A tensor of states.
            actions (torch.Tensor): A tensor ofactions.

        Returns:
            torch.Tensor: A tensor of the estimated rewards.
        """
        estimated_rewards = [
            self._estimate_reward_with_single_estimator(reward_estimator, states, actions)
            for reward_estimator in self._reward_estimators
        ]
        return torch.mean(torch.stack(estimated_rewards), dim=0)

    def _trajectory_reward(self, trajectory) -> float:
        """Calculates the real reward of a trajectory by summing up the trajectories provided reward values.

        Args:
            trajectory (lsit): List of steps, with each step being a tuple of state, action, and reward.

        Returns:
            float: The real reward of the trajectory.
        """
        return sum(step[2] for step in trajectory)
                
    def _trajectory_estimated_reward(self, reward_estimator, trajectory) -> torch.Tensor:
        """Estimates the reward of a trajectory only using a single reward estimator network

        Args:
            reward_estimator (RewardNetwork): The reward estimator to use.
            trajectory (lsit): List of steps, with each step being a tuple of state, action, and reward.

        Returns:
            torchTensor: The estimated reward of the trajectory.
        """
        states, actions, _ = zip(*trajectory)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        estimated_reward = torch.sum(self._estimate_reward_with_single_estimator(reward_estimator, states, actions))
        return estimated_reward
    
    def _simulate_human_feedback(self, trajectory1, trajectory2, random_feedback_chance: float = 0.1) -> float:
        """Uses the three valued feedback function to simulate a human feedback on which trajectory is preferred between two trajectories.

        Args:
            trajectory1 (list): List of steps, with each step being a tuple of state, action, and reward.
            trajectory2 (list): List of steps, with each step being a tuple of state, action, and reward.
            random_feedback_chance (float, optional): Chance to return a random feedback, used to better simulate human mistakes. Defaults to 0.1.

        Returns:
            float: Returns a feedback label. 0, 1, or 0.5
                0: The first trajectory is preferred
                1: The second trajectory is preferred
                0.5: It is uncertain which trajectory is preferred
        """       
        # a default 10 % chance to return a random feedback
        if random.random() < random_feedback_chance:
            return random.choice([0, 1, 0.5])
        
        # Get the rewards of the trajectories
        trajectory_reward1 = self._trajectory_reward(trajectory1)
        trajectory_reward2 = self._trajectory_reward(trajectory2)
            
        # Get the probabilities of the human feedback
        preferred_probs = ff.three_valued_feedback_function(trajectory_reward1, trajectory_reward2, eps=self._UNCERTAINTY_REGION_EPSILON)
            
        # Choose one feedback based on the probabilities
        preferred = torch.multinomial(preferred_probs, 1).item()
        
        # Convert the index to a feedback label
        if preferred == 2:
            preferred = 0.5
        return preferred
    
    def _select_trajectories_for_queries(self, amount: int = 1, sampling_amount_multipler: int = 20, top_amount_multipler: int = 3) -> list:
        """Selects trajectories from the trajectory memory based on their variance between the estimated rewards of the reard estimator networks. 
        At most about half the trajectories will be retuned.

        Args:
            amount (int, optional): The amount of feedbacks intended to collect, is used as a base for the sampling and cut off amount. Defaults to 1.
            sampling_amount_multipler (int, optional): The multiplier for the amount for the sampling of the trajectories from the trajectory memory. Defaults to 20.
            top_amount_multipler (int, optional): The multiplier for the amount for the number of the trajectories returned based on their variance. Defaults to 3.

        Raises:
            RuntimeError: If there are less than four trajectories in the trajectory memory

        Returns:
            list: A list of the trajectories with the highest variances between the reward estimators, sorted by the variances. Will return at most half of the trajectories in memory.
        """
        if amount < 1:
            raise ValueError("Amount of desired feedbacks must be bigger than 0!")
        
        if sampling_amount_multipler < 1:
            raise ValueError("The sampling amount multiplier must be bigger than 0!")
        
        if top_amount_multipler < 1:
            raise ValueError("The top amount multiplier must be bigger than 0!")

        if len(self._trajectory_memory) < 4:
            raise RuntimeError("There needs to be at least four trajectories to optimize the reward!")
        
        # Sampling a large amount of trajectories from the trajectory memory
        trajectories = random.sample(self._trajectory_memory, min(len(self._trajectory_memory), amount * sampling_amount_multipler))
        
        # Calculate variances between estimated rewards for each trajectory
        if len(self._reward_estimators) > 1:
            variances = torch.tensor([
                torch.var(torch.stack([self._trajectory_estimated_reward(estimator, traj) for estimator in self._reward_estimators]))
                for traj in trajectories
            ])
        # Use dummy variances if there is only one reward estimator
        else:
            variances = torch.zeros(len(trajectories))
                                    
        # Select top trajectories based on the highest variances
        top_indices = torch.topk(variances, min(round(len(trajectories)/2), amount * top_amount_multipler)).indices
        return [trajectories[i] for i in top_indices]
    
    def _collect_trajectory_feedback(self, amount: int = 1) -> None:
        """Collecting simulated human feedbacks, by selecting trajectories and simulating human feedback for them, then saving the feedback into the feedback memory.

        Args:
            amount (int, optional): The amount of simulated human feedbacks to collect and save. Should not be bigger than half the number of trajectories in provided. Defaults to 1.
        """
        # Selecting at most about half of the trajectories with the most variance between the estimated rewards
        trajectories = self._select_trajectories_for_queries(amount)
        
        for _ in range(amount):
            trajectory1, trajectory2 = random.sample(trajectories, 2)
                                                    
            # Get simulated human feedback (using actual rewards), with no random feedback
            feedback = self._simulate_human_feedback(trajectory1, trajectory2, random_feedback_chance=0.0)
                
            # Save simulated feedback into feedback memory
            self._feedback_memory.append((feedback, trajectory1, trajectory2))

    def _calculate_loss(self, reward_estimator) -> tuple[float, list]:
        """Calulated the cross entropy loss between the estimated feedbacks and the stored simulated human feedbacks simulated feedbacks.

        Args:
            reward_estimator (RewardNetwork): one of the reward estimators used in this class

        Returns:
            average_loss (float): The loss of one reward estimator.
            labels_counter (list): A list with the amount of times each label was used during loss calcuation. The labels are orderd as 0, 1, 0.5.
        """        
        loss = 0
        labels_counter = [0, 0, 0]
            
        # Get len(feedback_memory) random entries with replacement from feedback_memory
        random_trajectory_feedbacks = random.choices(self._feedback_memory, k=len(self._feedback_memory))
        for feedback, trajectory1, trajectory2 in random_trajectory_feedbacks:
                                
            # Calculate estimated rewards for both trajectories
            estimated_reward1 = self._trajectory_estimated_reward(reward_estimator, trajectory1)
            estimated_reward2 = self._trajectory_estimated_reward(reward_estimator, trajectory2)
                                    
            # Calculate feedback probabilities based on estimated rewards
            probs = ff.three_valued_feedback_function(estimated_reward1, estimated_reward2, eps=self._UNCERTAINTY_REGION_EPSILON)

            # Ensure no zero probabilities
            probs = torch.clamp(probs, min=self._MIN_POSITIVE_FLOAT32)

            # Calculate loss based on the formula in the paper "Deep Reinforcement Learning from Human Preferences" by Christiano and Leike et al.
            if feedback == 0:
                loss -= torch.log(probs[0])
                labels_counter[0] += 1
            elif feedback == 1:
                loss -= torch.log(probs[1])
                labels_counter[1] += 1
            else:  # feedback == 0.5
                loss -= torch.log(probs[2])
                labels_counter[2] += 1
            
        return loss, labels_counter
    
    def optimize_reward(self, query_amount: int = 1) -> tuple[float, list]:
        """Optimizes all reward esimator networks using the cross entropy loss between the estimated feedbacks and the stored simulated human feedbacks simulated feedbacks.

        Parameters:
            query_amount (int, optional): The amount of simulated human feedbacks to collect and save. Defaults to 1.

        Returns:
            average_loss (float): The average loss of all the reward estimators.
            labels_counter (list): A list with the average amount of times each label was used during this optimization. The labels are orderd as 0, 1, 0.5.
        """
        if query_amount < 1:
            raise ValueError("The query amount must be greater than 0!")

        # Collecting simulated human feedbacks, these are saved in feedback memory to also use later
        with torch.no_grad():
            self._collect_trajectory_feedback(query_amount)

        # Calculating the average loss of all reward estimators and the average occurence of the feedback lables
        average_loss = 0
        average_labels_counter = [0, 0, 0]
        for reward_estimator, reward_estimator_optimizer in zip(self._reward_estimators, self._reward_estimator_optimizers):
            loss, labels_counter = self._calculate_loss(reward_estimator)
            
            # Backpropagate and updating each reward estimator
            reward_estimator_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_estimator.parameters(), self._CLIP_GRAD_NORM_MAX_NORM)
            reward_estimator_optimizer.step()
            
            # Sum up the loss and label countes
            average_loss += loss.item()
            average_labels_counter = [average_labels_counter[i] + labels_counter[i] for i in range(len(labels_counter))]

        # Average the loss and labels counter
        average_loss /= len(self._reward_estimators)
        labels_counter = [x / len(self._reward_estimators) for x in labels_counter]
        return average_loss, labels_counter
