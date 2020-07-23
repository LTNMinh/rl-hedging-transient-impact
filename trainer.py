from typing import Dict, Tuple, Sequence,List
from copy import deepcopy

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
        self.buffer = collections.deque(maxlen = capacity)
        self.scaler = StandardScaler()
    
    def __len__(self) -> None:
        return len(self.buffer)
    
    def append(self,experience: Experience) -> None:
        self.buffer.append(experience)
    
    def standar_scaler(self) -> None: 
        states,actions,rewards,next_states,dones = zip(*self.buffer) 
        self.scaler.fit_transform(states) 
        return self.scaler 

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

