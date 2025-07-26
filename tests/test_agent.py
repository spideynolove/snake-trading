import unittest
import numpy as np
import torch
from core.agent import Agent
from env.forex_env import ForexEnv
import pandas as pd

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()
        
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'open': np.random.uniform(1.2000, 1.2100, 100),
            'high': np.random.uniform(1.2000, 1.2100, 100),
            'low': np.random.uniform(1.2000, 1.2100, 100),
            'close': np.random.uniform(1.2000, 1.2100, 100),
            'volume': np.random.randint(1000, 5000, 100)
        })
        self.env = ForexEnv(sample_data)
    
    def test_agent_initialization(self):
        self.assertEqual(self.agent.n_games, 0)
        self.assertEqual(self.agent.epsilon, 0)
        self.assertEqual(self.agent.gamma, 0.9)
        self.assertIsNotNone(self.agent.model)
        self.assertIsNotNone(self.agent.trainer)
        self.assertEqual(len(self.agent.memory), 0)
    
    def test_get_state(self):
        state = self.agent.get_state(self.env)
        self.assertEqual(len(state), 4)
        self.assertTrue(isinstance(state, np.ndarray))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in state))
    
    def test_get_action_random(self):
        self.agent.epsilon = 100
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = self.agent.get_action(state)
        
        self.assertEqual(len(action), 3)
        self.assertEqual(sum(action), 1)
        self.assertTrue(any(action))
    
    def test_get_action_model(self):
        self.agent.epsilon = 0
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = self.agent.get_action(state)
        
        self.assertEqual(len(action), 3)
        self.assertEqual(sum(action), 1)
        self.assertTrue(any(action))
    
    def test_remember(self):
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = [1, 0, 0]
        reward = 10
        next_state = np.array([0.2, 0.3, 0.4, 0.5])
        done = False
        
        initial_memory_size = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.memory), initial_memory_size + 1)
        
        stored_experience = self.agent.memory[-1]
        self.assertTrue(np.array_equal(stored_experience[0], state))
        self.assertEqual(stored_experience[1], action)
        self.assertEqual(stored_experience[2], reward)
        self.assertTrue(np.array_equal(stored_experience[3], next_state))
        self.assertEqual(stored_experience[4], done)
    
    def test_train_short_memory(self):
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = [1, 0, 0]
        reward = 10
        next_state = np.array([0.2, 0.3, 0.4, 0.5])
        done = False
        
        try:
            self.agent.train_short_memory(state, action, reward, next_state, done)
        except Exception as e:
            self.fail(f"train_short_memory raised an exception: {e}")
    
    def test_train_long_memory_empty(self):
        try:
            self.agent.train_long_memory()
        except Exception as e:
            self.fail(f"train_long_memory with empty memory raised an exception: {e}")
    
    def test_train_long_memory_with_data(self):
        for i in range(10):
            state = np.random.rand(4)
            action = [0, 0, 0]
            action[np.random.randint(3)] = 1
            reward = np.random.randint(-10, 11)
            next_state = np.random.rand(4)
            done = np.random.choice([True, False])
            
            self.agent.remember(state, action, reward, next_state, done)
        
        try:
            self.agent.train_long_memory()
        except Exception as e:
            self.fail(f"train_long_memory with data raised an exception: {e}")
    
    def test_model_consistency(self):
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        with torch.no_grad():
            self.agent.epsilon = 0
            action1 = self.agent.get_action(state)
            action2 = self.agent.get_action(state)
            
            self.assertEqual(action1, action2)
    
    def test_epsilon_decay_behavior(self):
        initial_epsilon = self.agent.epsilon
        self.agent.n_games = 50
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = self.agent.get_action(state)
        
        expected_epsilon = 80 - 50
        self.assertEqual(expected_epsilon, 30)
    
    def test_memory_maxlen(self):
        from core.agent import MAX_MEMORY
        
        for i in range(MAX_MEMORY + 100):
            state = np.random.rand(4)
            action = [0, 0, 0]
            action[np.random.randint(3)] = 1
            reward = np.random.randint(-10, 11)
            next_state = np.random.rand(4)
            done = False
            
            self.agent.remember(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.memory), MAX_MEMORY)

if __name__ == '__main__':
    unittest.main()