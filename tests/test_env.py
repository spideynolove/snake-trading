import unittest
import numpy as np
import pandas as pd
from env.forex_env import ForexEnv

class TestForexEnv(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'open': np.linspace(1.2000, 1.2100, 100),
            'high': np.linspace(1.2010, 1.2110, 100),
            'low': np.linspace(1.1990, 1.2090, 100),
            'close': np.linspace(1.2005, 1.2105, 100),
            'volume': np.random.randint(1000, 5000, 100)
        })
        self.env = ForexEnv(self.sample_data)
    
    def test_env_initialization(self):
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.balance, 10000.0)
        self.assertIsNone(self.env.position)
        self.assertEqual(len(self.env.trades_history), 0)
        self.assertEqual(self.env.portfolio_value, 10000.0)
    
    def test_reset(self):
        self.env.current_step = 50
        self.env.balance = 9000.0
        self.env.portfolio_value = 9000.0
        self.env.trades_history = [{'test': 'data'}]
        
        self.env.reset()
        
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.balance, 10000.0)
        self.assertIsNone(self.env.position)
        self.assertEqual(len(self.env.trades_history), 0)
        self.assertEqual(self.env.portfolio_value, 10000.0)
    
    def test_get_current_price(self):
        price = self.env.get_current_price()
        expected_price = self.sample_data.iloc[0]['close']
        self.assertEqual(price, expected_price)
        
        self.env.current_step = 10
        price = self.env.get_current_price()
        expected_price = self.sample_data.iloc[10]['close']
        self.assertEqual(price, expected_price)
    
    def test_get_state_shape(self):
        state = self.env.get_state()
        self.assertEqual(len(state), 4)
        self.assertTrue(isinstance(state, np.ndarray))
    
    def test_get_state_no_position(self):
        state = self.env.get_state()
        
        self.assertTrue(-1 <= state[0] <= 1)
        self.assertEqual(state[1], 0.0)
        self.assertEqual(state[2], 0.0)
        self.assertTrue(0 <= state[3] <= 1)
    
    def test_step_action_1_long_entry(self):
        initial_balance = self.env.balance
        reward, done, portfolio_value = self.env.step(1)
        
        self.assertIsNotNone(self.env.position)
        self.assertEqual(self.env.position.direction, 1)
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(self.env.balance, initial_balance)
    
    def test_step_action_2_short_entry(self):
        initial_balance = self.env.balance
        reward, done, portfolio_value = self.env.step(2)
        
        self.assertIsNotNone(self.env.position)
        self.assertEqual(self.env.position.direction, -1)
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(self.env.balance, initial_balance)
    
    def test_step_action_0_no_position(self):
        initial_balance = self.env.balance
        reward, done, portfolio_value = self.env.step(0)
        
        self.assertIsNone(self.env.position)
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(self.env.balance, initial_balance)
    
    def test_long_position_profitable_close(self):
        self.env.step(1)
        
        self.env.current_step = 10
        initial_balance = self.env.balance
        entry_price = self.env.position.entry_price
        current_price = self.env.get_current_price()
        
        self.assertGreater(current_price, entry_price)
        
        reward, done, portfolio_value = self.env.step(0)
        
        self.assertIsNone(self.env.position)
        self.assertEqual(reward, 10)
        self.assertGreater(self.env.balance, initial_balance)
        self.assertEqual(len(self.env.trades_history), 1)
    
    def test_short_position_profitable_close(self):
        self.env.current_step = 50
        self.env.step(2)
        
        entry_price = self.env.position.entry_price
        
        self.env.current_step = 10
        current_price = self.env.get_current_price()
        
        self.assertLess(current_price, entry_price)
        
        initial_balance = self.env.balance
        reward, done, portfolio_value = self.env.step(0)
        
        self.assertIsNone(self.env.position)
        self.assertEqual(reward, 10)
        self.assertGreater(self.env.balance, initial_balance)
        self.assertEqual(len(self.env.trades_history), 1)
    
    def test_long_position_losing_close(self):
        self.env.current_step = 50
        self.env.step(1)
        
        entry_price = self.env.position.entry_price
        
        self.env.current_step = 10
        current_price = self.env.get_current_price()
        
        self.assertLess(current_price, entry_price)
        
        initial_balance = self.env.balance
        reward, done, portfolio_value = self.env.step(0)
        
        self.assertIsNone(self.env.position)
        self.assertEqual(reward, -10)
        self.assertLess(self.env.balance, initial_balance)
        self.assertEqual(len(self.env.trades_history), 1)
    
    def test_episode_termination(self):
        self.env.current_step = len(self.sample_data) - 2
        reward, done, portfolio_value = self.env.step(1)
        
        self.assertTrue(done)
    
    def test_position_state_with_long(self):
        self.env.step(1)
        state = self.env.get_state()
        
        self.assertEqual(state[1], 1.0)
    
    def test_position_state_with_short(self):
        self.env.step(2)
        state = self.env.get_state()
        
        self.assertEqual(state[1], -1.0)
    
    def test_unrealized_pnl_calculation(self):
        self.env.step(1)
        
        self.env.current_step = 10
        state = self.env.get_state()
        
        self.assertNotEqual(state[2], 0.0)
        self.assertTrue(-1 <= state[2] <= 1)
    
    def test_trades_history_structure(self):
        self.env.step(1)
        self.env.current_step = 10
        self.env.step(0)
        
        self.assertEqual(len(self.env.trades_history), 1)
        trade = self.env.trades_history[0]
        
        required_keys = ['entry_price', 'exit_price', 'direction', 'pnl', 'entry_step', 'exit_step']
        for key in required_keys:
            self.assertIn(key, trade)
    
    def test_multiple_positions_not_allowed(self):
        self.env.step(1)
        
        initial_position = self.env.position
        self.env.step(2)
        
        self.assertEqual(self.env.position, initial_position)
    
    def test_temporal_constraints_integration(self):
        test_data_point = {
            'timestamp': pd.Timestamp('2023-01-01 12:00:00'),
            'open': 1.2050,
            'high': 1.2060,
            'low': 1.2040,
            'close': 1.2055,
            'volume': 2000
        }
        
        self.env.temporal_constraints.set_current_data(test_data_point)
        
        price = self.env.get_current_price()
        self.assertEqual(price, 1.2055)

if __name__ == '__main__':
    unittest.main()