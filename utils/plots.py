import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

class TradingPlots:
    def __init__(self, save_dir="./plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_equity_curve(self, balance_history, title="Equity Curve", save_name=None):
        plt.figure(figsize=(12, 6))
        plt.plot(balance_history, linewidth=2)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Balance ($)")
        plt.grid(True, alpha=0.3)
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_drawdown(self, balance_history, title="Drawdown Analysis", save_name=None):
        balance_series = pd.Series(balance_history)
        peak = balance_series.expanding().max()
        drawdown = (balance_series - peak) / peak * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(balance_history, linewidth=2, label='Balance')
        ax1.plot(peak, linewidth=1, alpha=0.7, label='Peak')
        ax1.set_title("Balance vs Peak")
        ax1.set_ylabel("Balance ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax2.plot(drawdown, color='red', linewidth=1)
        ax2.set_title("Drawdown %")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_trade_analysis(self, trades, title="Trade Analysis", save_name=None):
        if not trades:
            return
        
        profits = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1.hist([t['pnl'] for t in trades], bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title("PnL Distribution")
        ax1.set_xlabel("PnL")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)
        
        if profits and losses:
            ax2.hist([profits, losses], bins=20, alpha=0.7, 
                    label=['Profits', 'Losses'], color=['green', 'red'])
            ax2.set_title("Profit vs Loss Distribution")
            ax2.set_xlabel("PnL")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        durations = [t['exit_step'] - t['entry_step'] for t in trades]
        ax3.hist(durations, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_title("Trade Duration Distribution")
        ax3.set_xlabel("Duration (steps)")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)
        
        cumulative_pnl = np.cumsum([t['pnl'] for t in trades])
        ax4.plot(cumulative_pnl, linewidth=2)
        ax4.set_title("Cumulative PnL")
        ax4.set_xlabel("Trade Number")
        ax4.set_ylabel("Cumulative PnL")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_training_progress(self, episode_data, title="Training Progress", save_name=None):
        episodes = [d['episode'] for d in episode_data]
        balances = [d['final_balance'] for d in episode_data]
        epsilons = [d['epsilon'] for d in episode_data]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(episodes, balances, linewidth=2, color='blue')
        ax1.set_title("Training Progress - Balance")
        ax1.set_ylabel("Final Balance ($)")
        ax1.grid(True, alpha=0.3)
        
        window = min(50, len(balances) // 10) if len(balances) > 10 else 1
        if window > 1:
            moving_avg = pd.Series(balances).rolling(window=window).mean()
            ax1.plot(episodes, moving_avg, linewidth=2, color='red', alpha=0.7, 
                    label=f'MA({window})')
            ax1.legend()
        
        ax2.plot(episodes, epsilons, linewidth=2, color='orange')
        ax2.set_title("Epsilon Decay")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Epsilon")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_action_distribution(self, actions, title="Action Distribution", save_name=None):
        action_names = ['Close', 'Long', 'Short']
        action_counts = [actions.count(i) for i in range(3)]
        
        plt.figure(figsize=(8, 6))
        plt.pie(action_counts, labels=action_names, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.close()