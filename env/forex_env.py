import numpy as np
import pandas as pd
from collections import namedtuple
import threading
from queue import Queue

Position = namedtuple('Position', 'direction entry_price entry_step')

class TemporalConstraints:
    def __init__(self):
        self.current_data_point = None
        self.data_queue = Queue(maxsize=1)
        self.synchronization_events = {
            'F1': threading.Event(),
            'F2': threading.Event(), 
            'F3': threading.Event()
        }
    
    def set_current_data(self, data_point):
        self.current_data_point = data_point
    
    def get_current_data_only(self):
        return self.current_data_point

class ForexEnv:
    def __init__(self, data, initial_balance=10000.0):
        self.data = data
        self.initial_balance = initial_balance
        self.temporal_constraints = TemporalConstraints()
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.trades_history = []
        self.portfolio_value = self.initial_balance

    def get_current_price(self):
        current_data = self.temporal_constraints.get_current_data_only()
        if current_data is not None:
            return current_data['close']
        return self.data.iloc[self.current_step]['close']

    def get_portfolio_value(self):
        return self.portfolio_value

    def get_state(self):
        current_data = self.temporal_constraints.get_current_data_only()
        if current_data is None:
            current_data = self.data.iloc[self.current_step]
        
        current_price = current_data['close']
        
        if self.current_step >= 5:
            price_5_ago = self.data.iloc[self.current_step - 5]['close']
            momentum = (current_price - price_5_ago) / price_5_ago
            momentum = np.clip(momentum, -0.05, 0.05) / 0.05
        else:
            momentum = 0.0
        
        # Feature 2: Position state (like snake body position)
        position_state = 0.0
        if self.position is not None:
            position_state = 1.0 if self.position.direction == 1 else -1.0
        
        # Feature 3: Unrealized PnL (like distance to food)
        unrealized_pnl = 0.0
        if self.position is not None:
            if self.position.direction == 1:  # long
                unrealized_pnl = (current_price - self.position.entry_price) / self.position.entry_price
            else:  # short
                unrealized_pnl = (self.position.entry_price - current_price) / self.position.entry_price
            unrealized_pnl = np.clip(unrealized_pnl, -0.2, 0.2) / 0.2  # normalize [-1, 1]
        
        # Feature 4: Time since last action (like snake hunger)
        time_factor = (self.current_step % 24) / 24.0  # session time
        
        return np.array([momentum, position_state, unrealized_pnl, time_factor], dtype=float)

    def step(self, action):
        reward = 0
        done = False
        
        current_data = self.temporal_constraints.get_current_data_only()
        if current_data is None:
            current_data = self.data.iloc[self.current_step]
        
        self.temporal_constraints.set_current_data(current_data)
        current_price = current_data['close']
        
        # Action 0: Close position (or hold if no position)
        # Action 1: Enter long (if no position) 
        # Action 2: Enter short (if no position)
        
        if action == 1 and self.position is None:  # Enter long
            self.position = Position(direction=1, entry_price=current_price, entry_step=self.current_step)
        elif action == 2 and self.position is None:  # Enter short  
            self.position = Position(direction=-1, entry_price=current_price, entry_step=self.current_step)
        elif action == 0 and self.position is not None:  # Close position
            # Calculate PnL
            if self.position.direction == 1:  # long
                pnl = current_price - self.position.entry_price
            else:  # short
                pnl = self.position.entry_price - current_price
                
            # Binary reward like Snake
            reward = 10 if pnl > 0 else -10
            
            # Update balance
            self.balance += pnl
            self.portfolio_value = self.balance
            
            # Record trade
            self.trades_history.append({
                'entry_price': self.position.entry_price,
                'exit_price': current_price,
                'direction': self.position.direction,
                'pnl': pnl,
                'entry_step': self.position.entry_step,
                'exit_step': self.current_step
            })
            
            self.position = None
            
        self.current_step += 1
        
        # Simple termination
        if self.current_step >= len(self.data) - 1:
            done = True
            # Close any open position at end
            if self.position is not None:
                if self.position.direction == 1:
                    pnl = current_price - self.position.entry_price
                else:
                    pnl = self.position.entry_price - current_price
                reward = 10 if pnl > 0 else -10
                self.balance += pnl
                self.portfolio_value = self.balance
                self.position = None
                
        return reward, done, self.get_portfolio_value()