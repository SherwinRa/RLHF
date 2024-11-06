import matplotlib.pyplot as plt
import torch

class Plotter:
    def __init__(self) -> None:
        plt.ion()
        self.episode_durations = []

    def plot_durations(self, new_episode_duration, show_result=False):
        plt.figure(1)
        self.episode_durations.append(new_episode_duration)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
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