import numpy as np
from utils.metrics import TradingMetrics

class TradingEvaluator:
    def __init__(self):
        self.metrics = TradingMetrics()
        
    def evaluate_episode(self, env, final_balance, initial_balance=10000):
        self.metrics.add_balance(final_balance)
        
        for trade in env.trades_history:
            self.metrics.add_trade(
                trade['entry_price'],
                trade['exit_price'], 
                trade['direction'],
                trade['entry_step'],
                trade['exit_step']
            )
        
        episode_return = (final_balance - initial_balance) / initial_balance
        self.metrics.add_episode_reward(episode_return)
        
        return {
            'final_balance': final_balance,
            'episode_return': episode_return,
            'total_trades': len(env.trades_history),
            'current_metrics': self.metrics.get_summary()
        }
    
    def evaluate_agent_performance(self, agent, env, num_episodes=100):
        episode_scores = []
        episode_balances = []
        
        for episode in range(num_episodes):
            env.reset()
            
            while True:
                state = agent.get_state(env)
                action = agent.get_action(state)
                
                if any(action):
                    action_index = action.index(1)
                    reward, done, portfolio_value = env.step(action_index)
                    
                    if done:
                        episode_scores.append(portfolio_value)
                        episode_balances.append(portfolio_value)
                        break
                else:
                    break
        
        return {
            'avg_balance': np.mean(episode_balances),
            'std_balance': np.std(episode_balances),
            'min_balance': np.min(episode_balances),
            'max_balance': np.max(episode_balances),
            'success_rate': sum(1 for b in episode_balances if b > 10000) / len(episode_balances),
            'all_balances': episode_balances
        }
    
    def get_performance_summary(self):
        return self.metrics.get_summary()
    
    def reset_metrics(self):
        self.metrics.reset()