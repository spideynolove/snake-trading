import numpy as np

class TradingMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.trades = []
        self.balance_history = []
        self.episode_rewards = []
    
    def add_trade(self, entry_price, exit_price, direction, entry_step, exit_step):
        pnl = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
        
        trade = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'pnl': pnl,
            'entry_step': entry_step,
            'exit_step': exit_step,
            'duration': exit_step - entry_step
        }
        self.trades.append(trade)
    
    def add_balance(self, balance):
        self.balance_history.append(balance)
    
    def add_episode_reward(self, reward):
        self.episode_rewards.append(reward)
    
    def get_win_rate(self):
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
        return winning_trades / len(self.trades)
    
    def get_avg_profit(self):
        if not self.trades:
            return 0.0
        
        profits = [trade['pnl'] for trade in self.trades if trade['pnl'] > 0]
        return np.mean(profits) if profits else 0.0
    
    def get_avg_loss(self):
        if not self.trades:
            return 0.0
        
        losses = [trade['pnl'] for trade in self.trades if trade['pnl'] < 0]
        return np.mean(losses) if losses else 0.0
    
    def get_max_drawdown(self):
        if len(self.balance_history) < 2:
            return 0.0
        
        peak = self.balance_history[0]
        max_dd = 0.0
        
        for balance in self.balance_history:
            if balance > peak:
                peak = balance
            
            drawdown = (peak - balance) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def get_sharpe_ratio(self):
        if len(self.episode_rewards) < 2:
            return 0.0
        
        returns = np.array(self.episode_rewards)
        return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
    
    def get_total_pnl(self):
        return sum(trade['pnl'] for trade in self.trades)
    
    def get_summary(self):
        return {
            'total_trades': len(self.trades),
            'win_rate': self.get_win_rate(),
            'avg_profit': self.get_avg_profit(),
            'avg_loss': self.get_avg_loss(),
            'total_pnl': self.get_total_pnl(),
            'max_drawdown': self.get_max_drawdown(),
            'sharpe_ratio': self.get_sharpe_ratio(),
            'final_balance': self.balance_history[-1] if self.balance_history else 0.0
        }