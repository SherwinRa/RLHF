# RLHF

### Current Questions
Currently I am using for the length of the trajectory simply the total length an episode took. For cartpole this ranges from around 10 to maximum 500. Should this be changed?

In the atari paper, the reward estimator formula only has one observation-action as input, but in practice they use 4 observation-actions. Should I include this?

### To Do
237: Calculating the loss with the atari paper formula in 2.2.3, currently I am not including the third case of sigma = 0.5, I intent to test if that would improve the algorithm/computation

216: save the simulated human feedback in memory and use samples of them for optimization, similar to the trajectories

34: test with more environments

replace temporary code/algorithms with rllib algorithms

53: add better ploting for better visualization

Improve and try different algorithms for the optimization of the policy and the reward estimator.

### current results interpretation
It seems to be working

### current output
see output.png