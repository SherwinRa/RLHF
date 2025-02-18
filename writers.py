import os
import numpy as np

class RewardWriter:
    """This writer class is used to write the reward and its average to a file. In the format of:\n
    `<reward> <average_reward>`
    """
    def __init__(self, filename: str):
        """
        Args:
            filename (str): The file name of the file to write to. Can include folder path and file extension.
        """
        if not filename:
            raise ValueError("Filename must not be empty!")
        
        self._filename = filename
        self._data = []
        self._average = []
        
    def add(self, reward: float, average_over: int = 50) -> None:
        """Add a new reward value to the data in that should be written into the file.

        Args:
            reward (float): The new reward value to add.
            average_over (int, optional): The number of past values to average over for the new average value. 
                Will add no average if set to 0 or less. Defaults to 50.
        """
        self._data.append(reward)
        
        if average_over > 0:
            self._average.append(np.mean(self._data[-average_over:]))
                            
    # Write the data to the file
    def write_to_file(self):
        """Writes the data into the file, for each entry in a line with spaces in between each value. 
        Will create the directories and file should they not exist.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self._filename), exist_ok=True)

        # Create or overwrite the file
        with open(self._filename, 'w') as f:
            for i in range(len(self._data)):
                f.write('{} {}\n'.format(self._data[i], self._average[i]))

class LabelWriter:
    """This writer class is used to print the labels to a file. In the format of:\n
    `<episode> <label1> <label2> <label3> <total> <label1_percentage> <label2_percentage> <label3_percentage>`
    """
    def __init__(self, filename):
        """
        Args:
            filename (str): The file name of the file to write to. Can include folder path and file extension.
        """
        if not filename:
            raise ValueError("Filename must not be empty!")

        self._filename = filename
        self._data = []
        
    def add(self, episode: int,  labels: list) -> None:
        """Add a new reward value to the data in that should be written into the file.

        Args:
            episode (int): The new episode value to add.
            labels (list): The new label values to add. In the form of: [label 1, label 2, label 0.5]
        """
        total = sum(labels)
        self._data.append((episode, *labels, total, *(label / total for label in labels)))
                            
    # Write the data to the file
    def write_to_file(self) -> None:
        """Writes the data into the file, for each entry in a line with spaces in between each value. 
        Will create the directories and file should they not exist.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self._filename), exist_ok=True)

        # Create or overwrite the file
        with open(self._filename, 'w') as f:
            # Write the data in self.data into the file with spaces in between
            for entry in self._data:
                f.write(' '.join(map(str, entry)) + '\n')