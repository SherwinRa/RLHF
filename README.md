# RLHF

### Current Questions
Currently I am using for the length of the trajectory simply the total length an episode took. For cartpole this ranges from around 10 to maximum 500. Should this be changed?
Yes, try limit trajectory to about 10 to 50.

In the atari paper, the reward estimator formula only has one observation-action as input, but in practice they use 4 observation-actions. Should I include this?
If necessary. For cartpole not needed because of velocity.

How to compare different results, when so much randomness is affecting results?

### To Do
Calculating the loss with the atari paper formula in 2.2.3, currently I am not including the third case of sigma = 0.5, I intent to test if that would improve the algorithm/computation

test with more environments

replace temporary code/algorithms with rllib algorithms

add better ploting for better visualization

Improve and try different algorithms for the optimization of the policy and the reward estimator.

Optimze the code to increase performance

### current results interpretation
It seems to be working, but the results are a bit inconsistent

### current output
see output_1.png
see output_2.png
see output_3.png
see output_4.png

each with declining eps values, and the last one using the atari paper's probability function.
the outputs are just the results of the first try.