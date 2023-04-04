# TD(0) with Walking Environment

In this task we are going to work with an environment, where a robot can only walk left or right. 

![Alt text](https://github.com/PrzemekSekula/gym-walking/blob/main/img/gym-walking.jpg?raw=true "Gym Walking")


## Environment information
### Actions
- 0: Left
- 1: Right
### Observations
State (position of the robot). Integer number. The left-most (terminal) state is 0, the right-most is len(states) - 1.
### Reward
+1 in the right terminal state, otherwise 0.
### Environments
- `env = gym.make(Walking5-v0)` - 5 non-terminal (7 total) states
- `env = gym.make(Walking7-v0)` - 7 non-terminal (9 total) states
- `env = gym.make(Walking9-v0)` - 9 non-terminal (11 total) states
### Rendering
You may render the environment with `env.render()`. It works as usual, but you may also use additional arguments to make rendering look more informative. `env.render(state_values)` will display state values on each non-terminal state. See `mc.py` or `td0.py` examples.

## Installation:


```bash
pip install git+https://github.com/PrzemekSekula/gym-walking.git
```

## TO DO:

### 0) Warm-up
Become familiar with the environment and create a simple code that makes the robot walk randomly, and renders (visualizes) its actions. This is already done in the `warm-up-solved.py` file, but try do to it yourself.

### 1) Task 1 - Policy Evaluation with Monte Carlo
Fill in the `mc.py` notebook to evaluate the policy using The Monte Carlo method. This is
already solved in the `mc-solved.py` file, yet try to do it yourself.

### 2) Task 2 - Policy Evaluation with TD(0)
Fill in the `td0.py` notebook to evaluate the policy using TD(0) method.

### 3) Task 3 - TD(0) parameters
Tamper with TD(0) parameters (`alpha` and `gamma`) in order to understand their impact. Try to find the parameters that give reasonable state-value estimates in a reasonable time (e.g. 100 episodes)



