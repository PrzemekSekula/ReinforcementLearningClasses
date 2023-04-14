import argparse
from agents import SarsaAgent, QLearningAgent

import gym

parser = argparse.ArgumentParser(description='Renders an environment using selected agent')

parser.add_argument('-e', '--env',
                    type=str,
                    default='CliffWalking-v0',
                    choices = ['CliffWalking-v0', 'Taxi-v3'],
                    help='Environment. One of CliffWalking-v0, Taxi-v3. Default: CliffWalking-v0'
                   )


parser.add_argument('-g', '--greedy',
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True,
                    help='Choose greedy action only. Default: True'
                   )



parser.add_argument('-a', '--agent',
                    type=str,
                    default='SARSA',
                    choices = ['QLearning', 'SARSA'],
                    help='Agent. One of QLearning, SARSA. Default: QLearning'
                   )


parser.add_argument('-f', '--filename',
                    type=str,
                    default=None,
                    help='Filename to load Q_est from. Default: None (it is choosen by the Agent then)'
                   )




def render_env(agent, env, epsilon=0.1, greedy=False):
    path = []
    state, info = env.reset()
    states = [state]
    total_reward = 0
    total_steps = 0
    done = False
    env.render()
    while not done:
        if greedy:
            action = agent.Q_est[state].argmax()
        else:
            action = agent.get_action(state)
            
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        total_steps += 1
        states.append(state)
        env.render()
    return states, total_reward, total_steps

if __name__ == "__main__":
    args = parser.parse_args()
    env = gym.make(args.env, render_mode = 'human')
    
    if args.agent == 'QLearning':
        agent  = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon=0.1)
    else:
        agent = SarsaAgent(env, alpha=0.1, gamma=0.95, epsilon=0.1)

    if args.filename is None:
        agent.load()
    else:
        agent.load(args.filename)
    states, total_reward, total_steps = render_env(agent, env, agent.epsilon, greedy=args.greedy)
    print (f'Reward {total_reward} achieved in {total_steps} steps.')
    print ('States: {}'.format(states))
    
    