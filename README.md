# Three-Valued Feedback for Reinforcement Learning with Predicted Human Feedback

This implementation is based on the paper "Deep Reinforcement Learning from Human Preferences" by Christiano and Leike et al.

I tried to expand on the two-valued feedback function used in the paper, by creating a three-vaued feedback function. These functions where then used to predict human feedback, with which a reward estimator would be trained. The policy would then be optimized with only the estimated reward gathered from this reward estimator.

## Installation and Requirements

In this program I am using:

- Python 3.12.7
- PyTorch 2.4.1
- Gymnasium 1.0.0

Should these be installed, the program should be able to run. For the plotting of graphs, I used some Unix commands (paste, columns and tr) to manipulated the dat files the program generates, and gnuplot to plot the graphs. 

# Usage

To run the program and see a live plotting, run:

```sh
$ make demo
```

or

```sh
$ python main.py demo
```

To run the RL program with the real reward instead of an estimated reward, and create the corresponding dat files

```sh
$ make run_real
```

To run the RL program with the two-valued feedback function 10 times, and create the corresponding dat files

```sh
$ make run_noeps
```

To run the RL program with various epsilon values 10 times each, and create the corresponding dat files

```sh
$ make run_eps
```

# Folders and Files

- dat folder: Contains the dat files generated from the program and the combined dat files used for the plotting.
- gnuplotscripts folder: Contains the gnuplot scripts used to generate the pdf plots from the dat files.
- plot folder: Contains the generated pdf plots.
- feedback_function.py: Has the two-valued and three-valued feedback functions.
- main.py: Instantiates and runs the RLHF class with different parameters.
- plotter.py: Is used to visualize live plotting of the running program.
- policy_network.py: Is the neural newtork used for the policy in the RLHF class.
- reward_estimatior.py: This class is in charge of managing the multiple reward estimator neural networks, optimzing them, selecting queries and predicting human feedback based on the feedback functions.
- reward_network.py: Is the neural newtork used for the reward estimators in the RewardEstimator class.
- rhlf.py: This class manages the Environment and the policy.
- writers.py: Has writer classes that generate the specific dat files.