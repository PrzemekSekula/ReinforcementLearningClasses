import gym 
import time
import pygame
import random
 
class Game:
  def __init__(self, fps=30):
    env = gym.make('CartPole-v1', render_mode = 'human')
    rewards=0
    state, info = env.reset() # init observation
    env.render()
    print ('Make sure that the game window is active\r', end='')
    time.sleep(2)
    print ('                                             \r', end='')
    done = False
    while not done:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
            print ('LEFT     \r', end='')
        elif keys[pygame.K_RIGHT]:
            print ('RIGHT    \r', end='')

            action = 1    
        else:        
            action = random.randint(0, 1)
            print ('Random     \r', end='')
            


        state, reward, done, truncated, info = env.step(action)
        rewards += reward
        time.sleep(1 / fps)
        env.render()
    env.close()
    print ('Total Reward:',rewards)


mygame=Game()