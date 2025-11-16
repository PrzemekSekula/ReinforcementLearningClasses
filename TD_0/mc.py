import argparse


import gymnasium as gym
import gym_walking
import numpy as np

parser = argparse.ArgumentParser(description='Runs Monte Carlo state value estimate on Walking5-v0.')

parser.add_argument('-e', '--episodes',
                    type=int,
                    default=10,
                    help='Number of episodes. Default: 10'
                   )

parser.add_argument('-g', '--gamma',
                    type=float,
                    default=1.0,
                    help='Gamma parameter. Update is based on gamma * (R - V). Default: 1.0'
                   )

parser.add_argument('-r', '--render',
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True,
                    help='Whether to render or not. Default: True'
                   )


env = gym.make('Walking5-v0')
pi = lambda x: np.random.randint(2) # uniform random policy

def mc(pi, env, gamma=1.0, n_episodes=10, render=True):
    # ENTER YOUR CODE HERE
    # You should return the vector of state values V
    V = None
    return V

if __name__ == "__main__":
    args = parser.parse_args()
    V = mc(
        pi, env, 
        gamma=args.gamma, 
        n_episodes=args.episodes,
        render=args.render
        )
    print ('Final state values: {}'.format(V))