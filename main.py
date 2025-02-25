import gymnasium as gym
import sys

from rlhf import RLHF
from writers import RewardWriter, LabelWriter


def main_eps(repetitions: int, eps_values: list = [0, 10, 100, 1000]) -> None:
    """Runs the RL program using only the real reward and not an estimated reward.

    Args:
        repetitions (int): The amount of how often the RL program is run for each value in eps_values.
        eps_values (list): The different epsilon values for the three-valued feedback that should be used when running the RL program.
    """
    for eps in eps_values:
        for i in range(repetitions):
            single_with_writing(f"main{eps}_{i}", epsilon=eps)
            
def main_no_eps(repetitions: int) -> None:
    """Runs the RL program using the two-valued feedback function instead of the three-valued feedback function.

    Args:
        repetitions (int): The amount of how often the RL program is run.
    """
    for i in range(repetitions):
        single_with_writing(f"mainnoeps_{i}", epsilon=-100)

def main_real_reward(repetitions: int) -> None:
    """Runs the RL program using only the real reward and not an estimated reward.

    Args:
        repetitions (int): The amount of how often the RL program is run.
    """
    for i in range(repetitions):
        single_with_writing(f"mainreal_{i}", real_reward=True)

def single_with_writing(filename: str, epsilon: float = -100, real_reward: bool = False) -> None:
    """This runs the RL program once, and writes its reward and feedback labels into dat files.

    Args:
        filename (str): The filename of the resulting dat files
        epsilon (float, optional): The epsilon value for the three-valued feedback. If set less then or -100, It directly uses the 
            two-valued feedback. Defaults to -100.
        real_reward (bool, optional): If the real reward should be used instead of an estimated reward. Defaults to False.
    """
    reward_writer = RewardWriter(f"dat/reward/{filename}.dat")
    if not real_reward:
        label_writer = LabelWriter(f"dat/labels/{filename}.dat")
    else:
        label_writer = None
        
    env_cartpole = gym.make("CartPole-v1")
    RLHF(env_cartpole, real_reward, epsilon, reward_interval=10, policy_lr=0.001, reward_lr=0.00015, total_episodes=600, max_steps=510, memroy_size=500, exploration_decay=0.96, reward_writer=reward_writer, label_writer=label_writer).run()
    
    reward_writer.write_to_file()
    if not real_reward:
        label_writer.write_to_file()

def single_with_live_plotting(epsilon: float = 10) -> None:
    """For testing and demos run the RL program with live plotting

    Args:
        epsilon (float, optional): The epsilon value for the three-valued feedback. Defaults to 10.
    """
    env_cartpole = gym.make("CartPole-v1")
    RLHF(env_cartpole, False, epsilon, reward_interval=10, policy_lr=0.001, reward_lr=0.00015, total_episodes=600, max_steps=510, memroy_size=500, exploration_decay=0.96, plotting=True).run()


if __name__ == "__main__":
    # Ensure there is at least one arguments passed
    if len(sys.argv) > 1:
        argument = sys.argv[1]
        
        # Set how often the RL program will be run to take the average of
        repetition = 10
        
        # Based on the argument run the RLHF class with different parameters
        if isinstance(argument, float):
            single_with_live_plotting(argument)
        elif argument == "real":
            main_real_reward(repetition)
        elif argument == "noeps":
            main_no_eps(repetition)
        elif argument == "eps":
            main_eps(repetition, [0, 10, 100, 1000])
        elif argument == "demo":
            single_with_live_plotting()
        else:
            print("Usage: python main.py [real/noeps/eps/single] or a float value")
    else:
        print("Usage: python main.py [real/noeps/eps/single] or a float value")