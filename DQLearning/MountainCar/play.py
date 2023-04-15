import gym 
import time
import pygame
import random
 
class Game:
  def __init__(self, fps=30):
    env = gym.make('MountainCar-v0', render_mode = 'human')
    rewards=0
    state, info = env.reset() # init observation
    env.render()
    print ('Make sure that the game window is active\r', end='')
    action = 1
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
            action = 2    
        else:
            action = 1
            print ('NOTHIG   \r', end='')
                        


        state, reward, done, truncated, info = env.step(action)
        rewards += reward
        time.sleep(1 / fps)
        env.render()
    env.close()
    print ('Total Reward:',rewards)


mygame=Game()