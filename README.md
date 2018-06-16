# Reinforcement-Learning
This project involves the implementation of WoLF-based (Win or Learn Fast) learning agents and it is implementing **WoLF Policy Hill Climbing.**

The basic idea for this project was to vary the learning rates for the agents to support convergence of the algorithm. The main idea behind this algorithm is learn quickly while losing and slowly while winning. The specific method for determining when the agent is winning is by comparing the current policy’s expected payoff with that of the average policy over time.

### Problem is: 
```
Two robots are operating in a factory to bring metal bars to two different production halls. The metal 
bars are dispensed in one place, only one bar can be picked up at a time, and each robot can only 
carry one bar at a time. Once a metal bar is picked up, a new one will appear at the dispenser with 
probability 0.5 every time step (every action taken corresponds to one time step). Each robot has 3
action choices. It can either try to pick up a metal bar, deliver it to the production hall, or wait. 
If it tries to pick up a metal bar, it will succeed with probability 0.5 (due to imprecisions in its 
programming) if there is a metal bar available and fail if there is none available. If it tries to 
deliver a metal bar to the production hall, it will succeed with probability 1 if it is holding a metal 
bar and fail otherwise. If it decides to wait it will stay in place. If both robots try to pick up a 
metal bar at the same time, they will both fail. Each robot receives a payoff of 4 if it successfully 
delivers a metal bar to the production hall and incurs a cost of 1 if it tries to pick up a metal bar 
or if it tries to deliver one to the production hall (reflecting the energy it uses up). The wait 
action does not incur a cost.
```

### For the following problem:
```
• Each robot has two possible states = it either has a bar (S1) or it doesn’t (S0). Each 
dispenser has two possible states = it either makes a bar (S1) or it doesn’t (S0) Each 
robot has three possible actions = pick up a bar (P), deliver the bar (D) or wait (W).
• The states can be summarized as lists of 3 elements like (x,y,z) – where x is 1 if Robot1 
has a bar and 0 if it doesn’t, y is 1 if Robot2 has a bar and 0 if it doesn’t and z is 1 if 
Dispenser has a bar and 0 if it doesn’t. These are the following decision making markov 
processes –
```
![alt text](https://github.com/yatharth1908/Reinforcement-Learning/blob/master/Markov%20Chains/dispenser_mc.png "Dispenser Decision Markov Chain")
```
N is the one action for dispenser meaning it is making a new bar.
```

![alt text](https://github.com/yatharth1908/Reinforcement-Learning/blob/master/Markov%20Chains/robot_mc.png "Robot Decision Markov Chain")

### Algorithm implementation in the problem
```
• This algorithm requires two learning rates δl and δw with δl > δw and they are used to update
agents policy depending upon if agent is winning or losing. If the agent is losing the larger 
value of delta is used and vice versa. I used δw = 0.0025 and δl = 0.01 in which the ration is 
equal to 4. "	The discount factor is equal to 0.8 and the algorithm runs for 15000 iterations.
• Mean Policy is another idea used for this algorithm which determines if the agent is winning or 
losing.
• First, action was selected using random approach exploration with probability π(s,a) and then 
Q values, Mean Policy and Policy for first robot was updated according to the algorithm from the 
reference and the final policies were converging.
```

### References

For this project I followed the algorithm derived in paper [Rational and Convergent Learning in 
Stochastic Games by Dr. M. Bowling and Dr. M. Veloso.](http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf)

