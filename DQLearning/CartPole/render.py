import gym
import argparse
from dqn_agent import Agent


parser = argparse.ArgumentParser(description='Renders an environment using a trained agent.')

parser.add_argument('-e', '--env',
                    type=str,
                    default='CartPole-v1',
                    choices = ['CartPole-v1', 'MountainCar-v0', 'LunarLander-v2'],
                    help='OpenAI gym environment. Default: CartPole-v1'
                   )

parser.add_argument('-c', '--checkpoint-path',
                    type=str,
                    default='checkpoint.pth',
                    help='Path to the saved neural network. Default: checkpoint.pth'
                   )

parser.add_argument('-v', '--verbose',
                    type=int,
                    default=0,
                    help='Verbosity level. Default: 0'
                   )

parser.add_argument('-m', '--max-steps',
                    type=int,
                    default=2000,
                    help='Maximum number of steps. Default: 2000'
                   )


if __name__ == "__main__":
    args = parser.parse_args()
    env = gym.make(args.env, render_mode = 'human')
    
    if args.verbose:    
        print('State shape: ', env.observation_space.shape)
        print('Number of actions: ', env.action_space.n)


    agent = Agent(
        state_size=env.observation_space.shape[0], 
        action_size=env.action_space.n)

    agent.load(args.checkpoint_path)

    state, info = env.reset()
    env.render()
    rewards = 0
    for i in range(args.max_steps):
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
        rewards += reward
        env.render()

        if done:
            break 
    print (f'Total steps: {i}. Total reward: {rewards}')
    env.close()