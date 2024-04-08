"""
Code that renders an environment using selected agent
Arguments:
-e --env: Environment. One of CliffWalking-v0, Taxi-v3. Default: CliffWalking-v0
-g --greedy: Choose greedy action only. Default: True
-a --agent: Agent. One of QLearning, SARSA. Default: QLearning
-f --filename: Filename to load Q_est from. Default: None (it is choosen by the Agent then)
"""

import argparse
import gym
from agents import SarsaAgent, QLearningAgent

parser = argparse.ArgumentParser(
    description='Renders an environment using selected agent')

parser.add_argument(
    '-e',
    '--env',
    type=str,
    default='CliffWalking-v0',
    choices=[
        'CliffWalking-v0',
        'Taxi-v3'],
    help='Environment. One of CliffWalking-v0, Taxi-v3. Default: CliffWalking-v0')


parser.add_argument('-g', '--greedy',
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True,
                    help='Choose greedy action only. Default: True'
                    )


parser.add_argument('-a', '--agent',
                    type=str,
                    default='SARSA',
                    choices=['QLearning', 'SARSA'],
                    help='Agent. One of QLearning, SARSA. Default: QLearning'
                    )


parser.add_argument(
    '-f',
    '--filename',
    type=str,
    default=None,
    help='Filename to load Q_est from. Default: None (it is choosen by the Agent then)')


def render_env(agent, env, greedy=False):
    """Renders one episode of the environment using the agent
    Args:
        agent (any): Instance of SarsaAgent or QLearningAgent
        env (gym.Env): Gym environment
        greedy (bool, optional): Choose greedy action only. Defaults to False.

    Returns:
        _type_: _description_
    """
    state, _ = env.reset()
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

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
        total_steps += 1
        states.append(state)
        env.render()
    return states, total_reward, total_steps


if __name__ == "__main__":
    args = parser.parse_args()
    environment = gym.make(args.env, render_mode='human')

    if args.agent == 'QLearning':
        my_agent = QLearningAgent(
            environment,
            alpha=0.1,
            gamma=0.95,
            epsilon=0.1)
    else:
        my_agent = SarsaAgent(environment, alpha=0.1, gamma=0.95, epsilon=0.1)

    if args.filename is None:
        my_agent.load()
    else:
        my_agent.load(args.filename)
    final_states, reward_sum, nr_steps = render_env(
        my_agent, environment, greedy=args.greedy)
    print(f'Reward {reward_sum} achieved in {nr_steps} steps.')
    print('States: {}'.format(final_states))
