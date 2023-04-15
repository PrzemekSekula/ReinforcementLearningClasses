import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Neural network that estimates Q values."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state. It equals
                the number of features in the network.
            action_size (int): Dimension of each action. It equals 
                the number of the network outputs
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        
        # TODO: Create a network with 2 hidden layers (fc1 and fc2 
        # nodes) and output layer with action_size nodes.
        # use nn.Linear to create the layers (no activation function)
        self.fc1 = None # ENTER YOUR CODE HERE
        self.fc2 = None # ENTER YOUR CODE HERE
        self.fc3 = None # ENTER YOUR CODE HERE
        
    def forward(self, state):
        """Build a network that maps state -> action values.
        Args:
            state (torch.Tensor): The state of the environment
        Returns:
            torch.Tensor: The action values
        """
        
        # TODO: Implement the forward pass. Use ReLU activation function
        # ENTER YOUR CODE HERE
        return None # ENTER YOUR CODE HERE
