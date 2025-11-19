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
    
    # State values
    V = np.array([0] + [0.5] * (env.observation_space.n - 2) + [0])
    
    # How many times the state value was updated. We set the initial
    # state values to 0.5, so let's assume that we have already visited
    # each state once. This is not true, but it is a good approximation
    n_visited = np.array([1] * env.observation_space.n)

    for t in range(n_episodes):
        state, info = env.reset()
        n_visited[state] += 1
        terminal = False
        
        # All visited states in this episode (except terminal state)
        visited_states = []
        
       
        # Rewards for each state-action pair
        # It this case reward can be positive only for the last state
        # thus there is no need to store it. Yet, we keep it for the 
        # sake of generality
        rewards = [] 
        
        if render:
            env.unwrapped.render(V)
            
        while not terminal:
            visited_states.append(state)
            action = pi(state)
            next_state, reward, terminal, truncated, info = env.step(action)
            rewards.append(reward)
            
            state = next_state
            n_visited[state] += 1
            
            if render:
                env.unwrapped.render(V)
        
        
        # Now the episode is done. Let's update the state values        
        G = 0
        for state, reward in zip (visited_states[::-1], rewards[::-1]):
            n_visited[state] += 1
            G = gamma * G + reward
            V[state] += (G - V[state]) / n_visited[state]
            
        # Note: In this code  we only update the state values after 
        # each episode. The policy remains unchanged. If you want to
        # implement e-solf Monte Carlo you should also update the 
        # policy after each episode.
            
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