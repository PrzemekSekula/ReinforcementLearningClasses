import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size,
                 buffer_size = int(1e5),
                 batch_size = 64,
                 gamma = 0.99,
                 lr = 5e-4,
                 update_every = 4,):
        """Initialize an Agent object.
        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): replay buffer size. Default: int(1e5)
            batch_size (int): minibatch size. Default: 64
            gamma (float): discount factor. Default: 0.99
            lr (float): learning rate. Default: 5e-4
            update_every (int): how often to update the network. Default: 1            
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every

        # Q-Network
        fc1, fc2 = 64, 16 # Size of the layers
        # TODO: Create a network with 2 hidden layers (fc1 and fc2 nodes)
        # Remember to use .to(device) to move the network to the GPU        
        self.qnetwork = None # ENTER YOUR CODE HERE
        
        # TODO: Create the optimizer (Adam with learning rate lr)
        self.optimizer = None # ENTER YOUR CODE HERE

        # TODO: Create a Replay Buffer
        self.memory = None # ENTER YOUR CODE HERE
        
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        
        # TODO: Save experience in replay memory
        pass # ENTER YOUR CODE HERE
        
        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork(next_states).detach().max(dim=1, keepdim=True)[0]
        # TODO: Compute Q targets for current states 
        # You can use Q_targets_next * (1 - dones) to ensure that the value of the terminal state is 0
        Q_targets = None # ENTER YOUR CODE HERE

        Q_expected = self.local_network(states).gather(1, actions)


        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def save(self, path = 'checkpoint.pth'):
        """Saves the network

        Args:
            path (str): path to save the network. Default is 'checkpoint.pth'
        """
        torch.save(self.qnetwork.state_dict(), path)
        
    def load(self, path = 'checkpoint.pth'):
        """Loads the network

        Args:
            path (str): path with the weights. Default is 'checkpoint.pth'
        """
        self.local_network.load_state_dict(torch.load(path))        
 
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.
    I took this implementation from the Uacity Deep Reinforcement Learning Nanodegree
    """

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Adds a new experience to memory.
        Args:
            state (array_like): current state
            action (int): action taken
            reward (float): reward received
            next_state (array_like): next state
            done (bool): whether the episode is done
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly samples a batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Returns the current size of the buffer"""
        return len(self.memory)