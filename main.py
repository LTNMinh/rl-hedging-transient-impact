import random 
import numpy as np
import torch 


from math import exp, log, sqrt
from scipy.stats import norm


import copy 
import warnings
warnings.filterwarnings("ignore")

from hedging_environment import OptionHedgingEnv
from agent import DDPGHedgingAgent


if __name__ == '__main__':
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # environment
    env = OptionHedgingEnv(maturity=1.0)

    
    num_frames = 50000
    memory_size = 100000
    batch_size = 256
    ou_noise_theta = 1.0
    ou_noise_sigma = 0.05
    initial_random_episode = 100

    agent = DDPGHedgingAgent(
        env, 
        memory_size, 
        batch_size,
        ou_noise_theta,
        ou_noise_sigma,
        initial_random_steps=initial_random_episode,
        tau = 0.2,
        gamma = 0.99,
        name_cases='Option Hedging'
    )

    agent.train(num_frames)