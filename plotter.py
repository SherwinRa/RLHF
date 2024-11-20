import matplotlib.pyplot as plt
import torch

class Plotter:
    def __init__(self, ignore_eps=False, eps=None, policy_lr=None, reward_lr=None, policy_reward_ratio=None) -> None:
        plt.ion()
        self.episode_durations = []
        self.eps = eps
        self.policy_lr = policy_lr
        self.reward_lr = reward_lr
        self.policy_reward_ratio = policy_reward_ratio
        self.ignore_eps = ignore_eps

    def plot_durations(self, new_episode_duration):
        plt.figure(1)
        self.episode_durations.append(new_episode_duration)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        
        plt.clf()
        title = ''
        
        # Add parameters to the title if they are provided
        if any(param is not None for param in [self.eps, self.policy_lr, self.reward_lr, self.policy_reward_ratio]):
            #title += '\n'
            if self.eps is not None:
                if self.ignore_eps:
                    title += f'ε=off  '
                else:
                    title += f'ε={self.eps:.2f}  '
            if self.policy_lr is not None:
                title += f'Policy LR={self.policy_lr:.2e}  '
            if self.reward_lr is not None:
                title += f'Reward LR={self.reward_lr:.2e}  '
            if self.policy_reward_ratio is not None:
                title += f'P/R Ratio={self.policy_reward_ratio}'
        
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(durations_t.numpy())
        
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # Pause a bit so that plots are updated
        
    def show(self):
        plt.ioff()
        plt.show()