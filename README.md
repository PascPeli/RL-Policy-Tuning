# Reinforcement Learning Policy Tuning by changing the Reward Function



## Introduction

In this Notebook, I will be using a simple 2D game environment and a Reinforcement Learning agent with different reward functions to evaluate the importance of the reward function to the actions of the agent and to the outcome of the game. The policy of an RL agent is what determines his behaviors. The reward signal, calculated by the reward function, is what determines the policy of an agent.

<img src=https://raw.githubusercontent.com/PascPeli/RL-Policy-Tuning/master/data/main.gif alt="Drawing" style="width: 1000px;"/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig.1 - Resaults of Default and Penalized Reward Functions

The game environment used is a simulation of the game "Catch" where fruits fall from the top of the screen and the player has to catch them in his basket. Every time he catches one fruit he is rewarded with 1 point and every time he misses one he gets a negative reward of -1.
In each step of the game, the player can take one of the three following actions, go left, stay in the same place, or go right. These three are going to be the action space from which our agent can choose each step.
For training the agent we are going to use Deep Q-Learning. The Neural Network used in this experiment is a Feed-Forward Network with 2 hidden layers.

The environment used is based on the code of [Eder Santana](https://edersantana.github.io/articles/keras_rl/) for the same game. He created a simple "Catch" environment and also training and testing scripts for the agent. From his post, we can see the agent performing after being trained.


## The Problem

As we can see in Fig.2 below, althought the agent is able to perform quite well on the specific task after 1000 epochs, he performs a lot of unnecessary movements. Movements that contribute little to nothing to the final result of the episode. In this particular environment, this might not pose a major problem since moving comes with no cost. There are, of course, other problems where the same principles apply, changing the position of the agent to be in a specific place in time to catch fruits or avoid obstacles, but extra moves are unwanted. For example, when training driving policies of an autonomous vehicle we would prefer to have as little position adjustments as possible to ensure a smooth driving experience while maintaining transportation safety. Therefore, a policy is required that is capable of minimizing the unnecessary movements while sustaining its success rate.
<img src=https://edersantana.github.io/articles/keras_rl/catch.gif alt="Drawing" style="width: 500px;"/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;  Fig.2 - Ref.: Eder Santana - KerasPlaysCatch 

## Hypothesis

The actions of the agent are determined by its policy which is created based on his attempt to gather the highest reward possible. We can assign negative reward (punishment) to actions that are undesirable and thus try to minimize their occurrence while still maintaining or even increasing the winning rate of the agent.

### Reward Functions

For this reason, in addition to the Default function, two new approaches were implemented in the Catch environment for the calculation of the reward value. Those new reward functions assign a negative reward on each action of the agent that changes its location, left or right. Another punishment is added for each movement that leads to a move against the "walls" of the environment. Such a movement occurs for example when the agent is on the rightmost position of the environment and the action he chooses is also to go right, leading him to stay in the same position after he "hits" the wall. Those "undesirable" behaviors are hereby denoted as "extra moves". The two new reward functions differ only in the time step in which they return a "meaningful"* reward value. 

#### I. Default Reward Function

While using the Default Reward Function (1), our agent receives a reward value by the end of the episode.
This reward is a constant value and is determined only by the outcome of the episode. As described above the default approach returns a value of 1 for every successful episode, fruit in the basket, and value of -1 if it was unsuccessful, the fruit gets lost. The reward value for each timestep other than the last of the episode is equal to 0. In this method, we can say that the rewards are sparse. This way we don't penalize extra moves or moves that don't change the location of the agent (against the wall). The agent might understand that those moves offer nothing eventually, but this might take longer training times. 

$$Reward_{default}(t,w)=\begin{cases}\ \,\ 0 & t < last\_step \\ \ \,\ 1 & t = last\_step,\ w= 1\\-1 & t = last\_step,\ w= 0\end{cases}  \quad\big(1\big)$$

Where <b>t</b> denotes the time-step of the episode, <b>w</b> is its outcome and <b>p</b> is the penalty applied to extra moves. In our case:
$$ t \in (0,last\_step]\ ,\quad  w =\begin{cases}0 & unsuccessful\\1 & successful\end{cases}\  ,\quad p = -0.04$$

#### II. Penalized Once Reward Function

In this approach, the mentality remains the same with the default function with the exception that we also apply a penalization to/of extra moves. This leads to rewards that vary depending on the actions taken during the episode. The function returns a value of 0 after each step except from the last one in which the cumulative amount of the punishments corresponding to the extra moves is added to the outcome of the episode (2). As they are only awarded after the episode has finished, the agent has to make the correlation between extra moves and punishments itself without explicitly knowing which action corresponds to which reward. The rewards, here, are still sparse, the same as with the default function.

$$Reward_{penalty\_once}\ (t,w)=\begin{cases}\ \,\ 0 & t < last\_step \\ \ \,\ 1+ExtraMoves*p & t = last\_step,\ w= 1\\-1+ExtraMoves*p & t = last\_step,\ w= 0\end{cases}  \quad\big(2\big)$$

#### III. Penalized Reward Function

Conversely, The Penalized reward function returns at each step a value which is calculated based on the number of extra moves made up to this point and at the last step, the reward of the episode's outcome is added (3). The reward changes every time the agent is performing an extra move. Thus, the agent can assign a reward to each of his actions and in that way, he learns more consistently and in fewer iterations that extra moves yield lower rewards while maintaining the notion that catching the fruit is rewarded highly. 

$$Reward_{penalty}(t,w)=\begin{cases}\ \,\ ExtraMoves(t)*p & t < last\_step \\ \ \,\ 1+ExtraMoves(t)*p & t = last\_step,\ w= 1\\-1+ExtraMoves(t)*p & t = last\_step,\ w= 0\end{cases}  \quad\big(3\big)$$


## Game Modes

To make the game more interesting and more challenging for our agent, the code of the game was enhanced three new game modes. Now we can choose between four different game modes that affect the way the fruit is "falling". The different possible options are:
1. Straight Free Fall (default) - The fruit is falling down in a straight line.
2. Diagonal Fall - The fruit makes two moves per step, one down and one left or right.  
3. Diagonal Slow Fall - The fruit makes one move per step, once down and once to the side, left or right. 
4. Random Fall - The fruit next move is randomly picked from the sets [0,1] (stay, move down) for the Y-axis and [-1,0,1] (left, stay, right) for the X-axis. This game mode makes the environment Non-Deterministic. 

---
\* Here, "meaningful" is used to differentiate the non-zero rewards from the zero rewards which are returned after each time-step in both the Default and the Penalized Once reward functions.
