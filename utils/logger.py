import os
import json
import pickle
from datetime import datetime
from pathlib import Path

class TradingLogger:
    def __init__(self, log_dir="./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_log = []
        self.model_dir = self.log_dir / "models"
        self.model_dir.mkdir(exist_ok=True)
        
    def log_episode(self, episode, final_balance, trades_count, epsilon, loss=None):
        entry = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'final_balance': final_balance,
            'trades_count': trades_count,
            'epsilon': epsilon,
            'loss': loss
        }
        self.episode_log.append(entry)
        
        if episode % 10 == 0:
            self._save_episode_log()
    
    def log_trade(self, trade_data):
        trade_log_path = self.log_dir / f"trades_{self.session_id}.json"
        
        with open(trade_log_path, 'a') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                **trade_data
            }, f)
            f.write('\n')
    
    def save_checkpoint(self, agent, episode, final_balance):
        checkpoint = {
            'episode': episode,
            'final_balance': final_balance,
            'model_state': agent.model.state_dict(),
            'agent_params': {
                'n_games': agent.n_games,
                'epsilon': agent.epsilon,
                'gamma': agent.gamma
            },
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.model_dir / f"checkpoint_episode_{episode}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        latest_path = self.model_dir / "latest_checkpoint.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, agent, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.model_dir / "latest_checkpoint.pkl"
        
        if not checkpoint_path.exists():
            return False
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        agent.model.load_state_dict(checkpoint['model_state'])
        agent.n_games = checkpoint['agent_params']['n_games']
        agent.epsilon = checkpoint['agent_params']['epsilon']
        agent.gamma = checkpoint['agent_params']['gamma']
        
        return True
    
    def log_performance_metrics(self, metrics):
        metrics_path = self.log_dir / f"metrics_{self.session_id}.json"
        
        with open(metrics_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                **metrics
            }, f, indent=2)
    
    def _save_episode_log(self):
        episode_log_path = self.log_dir / f"episodes_{self.session_id}.json"
        
        with open(episode_log_path, 'w') as f:
            json.dump(self.episode_log, f, indent=2)
    
    def get_session_summary(self):
        if not self.episode_log:
            return {}
        
        balances = [entry['final_balance'] for entry in self.episode_log]
        
        return {
            'session_id': self.session_id,
            'total_episodes': len(self.episode_log),
            'avg_balance': sum(balances) / len(balances),
            'max_balance': max(balances),
            'min_balance': min(balances),
            'final_balance': balances[-1] if balances else 0
        }