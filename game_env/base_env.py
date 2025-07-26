from abc import ABC, abstractmethod
import numpy as np

class BaseForexEnv(ABC):
    def __init__(self, data, initial_balance=10000.0):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def get_state(self):
        pass
    
    @abstractmethod
    def get_current_price(self):
        pass
    
    @abstractmethod
    def get_portfolio_value(self):
        pass
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
    
    @property
    def action_space(self):
        return 3
    
    @property 
    def observation_space(self):
        return 4