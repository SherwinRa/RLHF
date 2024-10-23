# RLHF

### current questions
75: should the real reward be used or the estimated reward to calculate the reward of the trajectory? In the atari paper the estimated reward is being used, I think. But in my opinion using the real reward should simulate the human feedback better.
106: the real reward is being used to optimze the policy, should the estimated reward be used?

### to do
137: Calculating the loss with the atari paper formula in 2.2.3, currently I am not including the third case of sigma = 0.5
optimze the code to optimze the policy while creating trajectories for the human feedback
test with more environments
replace temporary code/algorithms with rllib algorithms
add ploting for better visualization

### current results interpretation
neither the total reward is increasing nor is the MSE of the estimated reward decreasing. The optimizer I implemented is not working yet.

### current output
20/500:
Total reward with trained policy in 9 steps: 9.0
MSE for the estimated reward: 1.4762367373700493

40/500:
Total reward with trained policy in 10 steps: 10.0
MSE for the estimated reward: 1.5181339454494598

60/500:
Total reward with trained policy in 8 steps: 8.0
MSE for the estimated reward: 1.4339245863225976

80/500:
Total reward with trained policy in 9 steps: 9.0
MSE for the estimated reward: 1.4941960738910265

100/500:
Total reward with trained policy in 9 steps: 9.0
MSE for the estimated reward: 1.4859229463798853

120/500:
Total reward with trained policy in 10 steps: 10.0
MSE for the estimated reward: 1.5048986754250162

140/500:
Total reward with trained policy in 11 steps: 11.0
MSE for the estimated reward: 1.5302871978251291

160/500:
Total reward with trained policy in 11 steps: 11.0
MSE for the estimated reward: 1.5282366692042295

180/500:
Total reward with trained policy in 10 steps: 10.0
MSE for the estimated reward: 1.501953603989357

200/500:
Total reward with trained policy in 10 steps: 10.0
MSE for the estimated reward: 1.5215650160924645

220/500:
Total reward with trained policy in 10 steps: 10.0
MSE for the estimated reward: 1.5110853558430672

240/500:
Total reward with trained policy in 8 steps: 8.0
MSE for the estimated reward: 1.4440507007191141

260/500:
Total reward with trained policy in 9 steps: 9.0
MSE for the estimated reward: 1.4740228208039916

280/500:
Total reward with trained policy in 10 steps: 10.0
MSE for the estimated reward: 1.5072819641066597

300/500:
Total reward with trained policy in 10 steps: 10.0
MSE for the estimated reward: 1.505758029271116

320/500:
Total reward with trained policy in 9 steps: 9.0
MSE for the estimated reward: 1.4779968580432532

340/500:
Total reward with trained policy in 10 steps: 10.0
MSE for the estimated reward: 1.4996258749685076

360/500:
Total reward with trained policy in 9 steps: 9.0
MSE for the estimated reward: 1.4877572199934508

380/500:
Total reward with trained policy in 9 steps: 9.0
MSE for the estimated reward: 1.4784817715931693

400/500:
Total reward with trained policy in 9 steps: 9.0
MSE for the estimated reward: 1.4920320271127427

420/500:
Total reward with trained policy in 10 steps: 10.0
MSE for the estimated reward: 1.5125802082805309

440/500:
Total reward with trained policy in 10 steps: 10.0
MSE for the estimated reward: 1.5003758658299493

460/500:
Total reward with trained policy in 8 steps: 8.0
MSE for the estimated reward: 1.446291567173294

480/500:
Total reward with trained policy in 9 steps: 9.0
MSE for the estimated reward: 1.4751477296297262

500/500:
Total reward with trained policy in 10 steps: 10.0
MSE for the estimated reward: 1.504208050648708
finished!