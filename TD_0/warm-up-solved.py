import gymnasium as gym
import gym_walking
import numpy as np

env = gym.make('Walking5-v0')
pi = lambda x: np.random.randint(2) # uniform random policy

state, info = env.reset()
env.render()
terminal = False
while not terminal:
    action = pi(state)
    next_state, reward, terminal, truncated, info = env.step(action)
    print (f'State: {state}, action: {action}, next state: {next_state}, reward: {reward}, terminal: {terminal}')
    env.render()
    state = next_state