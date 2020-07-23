import numpy as np 
import random 

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch import optim 

import gym 
import wandb
import pkbar 
import sys
import copy 
from typing import Dict, Tuple, Sequence,List
from sklearn.preprocessing import StandardScaler  

import collections

from typing import Dict, Tuple, Sequence,List
from copy import deepcopy

from actor_critic import Actor, Critic

Experience = collections.namedtuple(
    'Experience', field_names = ['state', 'action', 'reward', 'new_state','done' ]
)

class ReplayBuffer: 
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None: 
        self.buffer    = collections.deque(maxlen = capacity)
        self.scaler = StandardScaler()
    
    def __len__(self) -> None:
        return len(self.buffer)
    
    def append(self,experience: Experience) -> None:
        self.buffer.append(experience)
    
    # def standar_scaler(self) -> None: 
    #     states,actions,rewards,next_states,dones = zip(*self.buffer) 
    #     self.scaler.fit_transform(states) 
    #     return self.scaler 

    def sample(self,batch_size: int,device: str = 'cpu') -> Tuple:
        indices = np.random.choice(len(self.buffer),batch_size,replace = False)
        states,actions,rewards,next_states,dones = zip(*[self.buffer[idx] for idx in indices])

        return (torch.FloatTensor(states),
                torch.FloatTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.IntTensor(dones))

class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2,):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state

class DDPGHedgingAgent:
    """DDPGAgent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_episode (int): initial random action steps
        noise (OUNoise): noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """
    def __init__(self,
                    env: gym.Env,
                    memory_size: int,
                    batch_size: int,
                    ou_noise_theta: float,
                    ou_noise_sigma: float,
                    gamma: float = 0.99,
                    tau: float = 5e-3,
                    initial_random_episode: int = 1e4,
                    name_cases = 'myproject'):
        """ Initialize. """
        
        # Logger 
        self.wandb = wandb.init(project=name_cases)
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.env = env
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_episode = initial_random_episode
                
        # noise
        self.noise = OUNoise(
            action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
        )

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(self.device)
        
        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer
        self.actor_optimizer  = optim.Adam(self.actor.parameters() , lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # transition to store in memory
        self.transition = list()
        
        # total steps count
        self.total_step = 0
        # mode: train / test
        self.is_test = False
        self.populate(self.initial_random_episode)
    
        
        
    
    def populate(self, eps: int = 100) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences

        Args:
            steps: number of random steps to populate the buffer with
        """
        
        if not self.is_test:
            print("Populate Replay Buffer... ")
            kbar       = pkbar.Kbar(target=eps, width=20)
            state = self.env.reset()
            
            for i in range(eps):
                while True: 
                    # Get action from sample space 
                    selected_action = self.env.action_space.sample()
                    # selected_action = 0 
                    noise = self.noise.sample()
                    selected_action = np.clip(selected_action + noise, -1.0, 1.0)
                    
                    next_state, reward, done, _ = self.env.step(selected_action)
                    self.transition = [state, selected_action ,reward, next_state, int(done)]
                    self.memory.append(Experience(*self.transition))
                    
                    state = next_state
                    if done:         
                        state = self.env.reset()
                        break 
                
                kbar.add(1)

            # self.scaler = self.memory.standar_scaler()
                
    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state_s = self.scaler.transform([state])
        selected_action = self.actor(torch.FloatTensor(state_s).to(self.device)).item()
        # add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        
        self.transition = [state, selected_action]
        return selected_action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)
        
        if not self.is_test:
            self.transition += [reward, next_state, int(done)]
            self.memory.append(Experience(*self.transition))
    
        return next_state, reward, done
    
    def update_model(self) -> torch.Tensor:
        """ Update the model by gradient descent.
            Change the loss in to mean variance optimization
        """
        device = self.device  # for shortening the following lines
        
        state,action,reward,next_state,done = self.memory.sample(self.batch_size,self.device)
        
        state = torch.FloatTensor(self.scaler.transform(state)).to(device)
        next_state = torch.FloatTensor(self.scaler.transform(next_state)).to(device)
        # state = state.to(device)
        # next_state = next_state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        done = done.to(device)


        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value  = self.critic_target(next_state, next_action)
        curr_return = reward.reshape(-1,1) + self.gamma * next_value * masks.reshape(-1,1)
        
        # train critic
        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, curr_return)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.critic.parameters():
            p.requires_grad = False

        # train actor
        q_values = self.critic(state, self.actor(state))
        actor_loss = - q_values.mean() 
        # actor_loss = 0.5 * q_values.std() ** 2 
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for p in self.critic.parameters():
            p.requires_grad = True

        # target update
        self._target_soft_update()
        
        return actor_loss.data, critic_loss.data
    
    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        actor_losses = []
        critic_losses = []
        scores = []
        score = 0
        
        print("Training...")
        kbar       = pkbar.Kbar(target=num_frames, width=20)
        
        for self.total_step in range(1, num_frames + 1):
            action                   = self.select_action(state)
            next_state, reward, done = self.step(action)

            state  = next_state
            score += reward

            # if episode ends
            if done:         
                state = self.env.reset()
                scores.append(score)
                score = 0
                
                self._plot(self.total_step,scores, 
                            actor_losses, critic_losses,)

            # if training is ready
            if (len(self.memory) >= self.batch_size ): # and 
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            kbar.add(1)
                
        self.env.close()
        
    def test(self):
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        self.env.close()
        
        return score
    
    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
            
        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
    
    
    def _plot(self,  frame_idx: int, scores: List[float], 
                      actor_losses: List[float], critic_losses: List[float],):
        """Plot the training progresses."""
        
        self.wandb.log({'frame': frame_idx, 'score': scores[-1], 
                   'actor_loss': actor_losses[-1] , 'critic_loss': critic_losses[-1] })