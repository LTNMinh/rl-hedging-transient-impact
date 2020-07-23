import numpy as np
import gym
from gym import spaces

class OptionHedgingEnv(gym.Env):
    """
        Option Hedging Environment that follows gym interface

        Sell Call Option 
        Buy Underlying Assets 
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,sigma = 0.1,maturity = 100,time_interval = 0.1):
        super(OptionHedgingEnv, self).__init__()
        
        ############# Define space #############
        self.min_quantity = -1.0
        self.max_quantity = 1.0

        self.min_maturity = 0
        self.max_maturity = 1.0 

        self.min_price = 10 
        self.max_price = 200

        # self.low_state =  [self.min_price] * 252
        self.low_state = [self.min_price]
        # self.low_state.append(self.min_price)    # Strike 
        self.low_state.append(self.min_quantity) # Hodling 
        self.low_state.append(self.min_maturity) # Maturity 
        self.low_state = np.array(self.low_state)
        
        # self.high_state =  [self.max_price] * 252 
        self.high_state =  [self.max_price]
        # self.high_state.append(self.max_price)    # Strike 
        self.high_state.append(self.max_quantity)
        self.high_state.append(self.max_maturity)
        self.high_state = np.array(self.high_state)

        self.action_space =  spaces.Box(
                low = -self.max_quantity,
                high=  self.max_quantity,
                shape=(1,),
                dtype=np.float64
        )

        self.observation_space = spaces.Box(low  = self.low_state, 
                                            high = self.high_state,
                                            # shape=(1,3),  # Holding Assets, Assets Price, Time to Maturity 
                                            dtype=np.float64)
    
    def seed(self):
        """
            Random init condition on BS model
        """
        ################## Init Parameters ################
        #TODO: Random
        # Fix params 
        self.sigma = 0.3 
        self.mu = 0.1
        self.maturity = 1.0
        self.time_step = 252 
        self.dt = 1.0 / self.time_step

    def reset(self):
        """
            Reset the state of the environment to an initial state
            The quantity of holding asset is 0 
        """
        self.seed()

        start_quantity  = 0 
        start_price     = 100
        self.strike_price = start_price

        self.portfolio_value = np.zeros(252)
        self.state = np.array([start_price,
                                0,
                                self.maturity,])
        self.time_step = 0

        return self.state
        

    def step(self, action):
        """
            Underlying Price as Geometric Brownian Process 
            dS = mu * S_t * dt + sigma * S_t * dW 
        """
        try:
            action = action[0]
        except: 
            action = action
        
        price_path     = self.state[0]
        last_holding   = self.state[-2]
        time_to_mature = self.state[-1]

        # Update Price as Brownian Motion
        last_price     = price_path
        current_price  = last_price +\
                            self.mu  * last_price * self.dt +\
                            self.sigma * last_price * np.sqrt(self.dt) * np.random.randn()
        
        adjust_holding = last_holding + action

        self.portfolio_value[self.time_step] = adjust_holding * current_price
        # return_ = current_price / last_price - 1 
        # value_change = last_holding * return_
        # self.value_change.append(value_change)

        if self.time_step + 1 != 252: 
        # if max(self.dt,self.maturity) != self.dt:
            self.maturity +=  -self.dt 
            done   = False
            self.time_step +=  1 
            print(self.maturity,self.time_step)
            reward = adjust_holding * (current_price - last_price)
        else: 
            last_price = current_price
            current_price  = last_price +\
                            self.mu  * last_price * self.dt +\
                            self.sigma * last_price * np.sqrt(self.dt) * np.random.randn()
        
            # price_path.pop(0)
            # price_path.append(current_price)
            # return_ = current_price / last_price - 1 
            print(self.maturity,self.time_step)
            reward = 0
            # value_change  +=  adjust_holding * return_ 
            # self.value_change.append(adjust_holding * return_ )
            # value_change  +=  max(current_price - self.strike_price , 0) 
            # self.value_change.append(max(current_price - self.strike_price , 0) )
            # reward  =  -abs( value_change -  max(current_price - self.strike_price , 0))

            adjust_holding = 0 
            self.maturity  = 0 
            done           = True

        # reward = value_change - 0.2 * value_change ** 2
        # reward =  value_change - 0.5* (value_change - np.mean(self.value_change) ) ** 2 
        # price_path.extend([self.strike_price,adjust_holding,self.maturity])
        # self.state = np.array(price_path)
        # self.last_price = current_price
        self.state = np.array([
            current_price,
            adjust_holding,
            self.maturity,
            ])

        return self.state, reward, done, {}

    def render(self, mode='human', close=False):
        pass 