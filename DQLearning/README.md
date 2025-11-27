# Deep Q-Learning assignments
This folder contains various Deep Q-Learning assinments, including
- `00_VanillaDQL.ipynb` - A 'pure' algorithm without fixed targets. Used mainly for presentations
- `01-DQL_with_fixed_QTargets` - Real Deep Q-Learning algorithm. 
- `02-CNN_DuelingQN` - Deep Q-Learning for image-based states. It leverages the following improvements over the standard algorithm
	- Convolutional Neural Networks
	- Double Q-Learning
	- Duelling Q-Learning


# Installation

Everything is designed to work directly with Google Colab. If you want to install it locally, follow the procedure below.


1. Create an environment

```bash
conda create --name DQL python=3.13
```

2. Install Pytorch as described [here](https://pytorch.org/get-started/locally/). Pytorch installation depends on platform you are using.

3. Install other libraries 
```bash
# From ./DQLearning
conda activate DQL
pip install -r requirements.txt
```

