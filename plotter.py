import matplotlib.pyplot as plt
import torch

class Plotter:
    """This class plots up to four subplots while the program is running. Intended for visualizing the training process immediately.
    """
    def __init__(self, title=''):
        """
        Args:
            title (str, optional): The title of this plot. Defaults to ''.
        """
        self._values_dict = {}
        self._mean_dict = {}
        plt.ion()
        plt.title(title)
        
        # remove ticks of the main plot
        ax = plt.gca()
        ax.set_yticks([])
        ax.set_xticks([])
                
    def plot_value(self, value: float, subplot: int = 1, average_over: int = 50, xLabel: str = 'Episode', yLabel: str = 'Reward') -> None:
        """Add new value to the specific subplot and update that subplot.

        Args:
            value (float): The value to add to the subplot
            subplot (int, optional): The subplot index. Only four subplots are supported. 1-4. Defaults to 1.
            average_over (int, optional): The number of past values in this subplot to average over. Defaults to 50.
            xLabel (str, optional): The title of the x axis in this subplot. Defaults to 'Episode'.
            yLabel (str, optional): The title of the y axis in this subplot. Defaults to 'Reward'.
        """
        #check if subplot is valid 1-4
        if subplot < 1 or subplot > 4:
            raise ValueError("The subplot argument must be between 1 and 4, as this is a plot with four subplots.")
        
        # Create the lists for this subplot in the dictionaries if they don't exist
        if subplot not in self._values_dict:
            self._values_dict[subplot] = []
        if subplot not in self._mean_dict:
            self._mean_dict[subplot] = []
        
        # Add the value to the list in the dictionary
        self._values_dict[subplot].append(value)
        durations_t = torch.tensor(self._values_dict[subplot], dtype=torch.float)
        
        # Clear the subplot and plot the values
        ax = plt.subplot(220 + subplot)
        ax.cla()
        ax.plot(durations_t.numpy())

        # Plot the average of the last values if wanted
        if(average_over > 0):
            # Calculate, save and plot the average of the last values based on the average_over argument
            mean_length = min(len(durations_t), average_over)
            self._mean_dict[subplot].append(torch.mean(durations_t[-mean_length:]))
            mean_t = torch.tensor(self._mean_dict[subplot], dtype=torch.float)
            ax.plot(mean_t.numpy())

        # Set x and y labels
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        plt.pause(0.000001) # Pause a bit so that plots are updated
        
    def show(self) -> None:
        """Pause the plot and display it until the user closes the window. This method should be called after the plotting is finshed.
        """
        plt.ioff()
        plt.show()