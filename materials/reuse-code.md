# agent.py
import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

class Agent:
    def __init__(self, config_name='testing'):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=100000)
        self.batch_size = 1000
        self.model = Linear_QNet(input_size=4, output_size=2)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        if len(mini_sample) == 0:
            return
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model.load(filename)


# model.py
import torch
import torch.nn as nn
import torch.optim as optim
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size=4, output_size=2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            return True
        return False

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


# forex_game.py
import numpy as np
import pandas as pd

class ForexGameAI:
    def __init__(self, data, initial_balance=1000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.trades_history = []

    def get_portfolio_value(self):
        portfolio_value = self.balance
        if self.position is not None:
            current_price = self.data.iloc[self.current_step]['close']
            if self.position['direction'] == 1:
                pnl = (current_price - self.position['entry_price']) / self.position['entry_price']
            else:
                pnl = (self.position['entry_price'] - current_price) / self.position['entry_price']
            portfolio_value += self.position['size'] * pnl
        return portfolio_value

    def get_current_price(self):
        return self.data.iloc[self.current_step]['close']

    def get_state(self):
        if self.current_step == 0:
            price_momentum = 0
        else:
            price_momentum = (self.data.iloc[self.current_step]['close'] - self.data.iloc[self.current_step - 1]['close']) / self.data.iloc[self.current_step - 1]['close']
        position_state = 1 if self.position is not None and self.position['direction'] == 1 else (0.5 if self.position is not None else 0)
        unrealized_pnl = 0
        if self.position is not None:
            current_price = self.get_current_price()
            if self.position['direction'] == 1:
                unrealized_pnl = (current_price - self.position['entry_price']) / self.position['entry_price']
            else:
                unrealized_pnl = (self.position['entry_price'] - current_price) / self.position['entry_price']
        time_factor = self.current_step / len(self.data)
        return np.array([price_momentum, position_state, unrealized_pnl, time_factor], dtype=float)

    def play_step(self, action):
        initial_portfolio_value = self.get_portfolio_value()
        done = False
        reward = 0
        if self.position is not None:
            current_price = self.get_current_price()
            if self.position['direction'] == 1:
                pnl = (current_price - self.position['entry_price']) / self.position['entry_price']
            else:
                pnl = (self.position['entry_price'] - current_price) / self.position['entry_price']
            self.balance += self.position['size'] * pnl
            trade_record = {
                'entry_price': self.position['entry_price'],
                'exit_price': current_price,
                'direction': self.position['direction'],
                'pnl': pnl
            }
            self.trades_history.append(trade_record)
            self.position = None
        if action == 1:
            self.position = {
                'direction': 1,
                'entry_price': self.get_current_price(),
                'size': self.balance
            }
        elif action == 0:
            self.position = {
                'direction': 0,
                'entry_price': self.get_current_price(),
                'size': self.balance
            }
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        final_portfolio_value = self.get_portfolio_value()
        reward = final_portfolio_value - initial_portfolio_value
        return reward, done, final_portfolio_value


# real_data_loader.py
import pandas as pd
import numpy as np

def load_real_gbpusd_data(h1_path='/home/hung/ForexML/data/GBPUSD60.csv', start_samples=1000, total_samples=10000):
    try:
        h1_data = pd.read_csv(h1_path)
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in h1_data.columns for col in required_columns):
            raise ValueError(f"H1 data missing required columns. Found: {h1_data.columns.tolist()}")
        h1_data['time'] = pd.to_datetime(h1_data['time'], errors='coerce')
        h1_data = h1_data.dropna(subset=['time'])
        h1_data = h1_data.sort_values('time').reset_index(drop=True)
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            h1_data[col] = pd.to_numeric(h1_data[col], errors='coerce')
        h1_data = h1_data.dropna()
        h1_data = h1_data.iloc[start_samples:start_samples + total_samples].copy()
        if h1_data.empty:
            raise ValueError("No data left after applying start_samples and total_samples.")
        price_columns = ['open', 'high', 'low', 'close']
        volume_column = ['volume']
        h1_data[price_columns] = h1_data[price_columns].clip(lower=0.01)
        h1_data[volume_column] = h1_data[volume_column].clip(lower=1)
        return h1_data
    except Exception as e:
        print(f"Error loading GBPUSD H1 data: {e}")
        return pd.DataFrame()

def get_data_statistics(data):
    if data.empty:
        return {}
    stats = {}
    price_cols = ['open', 'high', 'low', 'close']
    stats['price_ranges'] = {
        'open': (data['open'].min(), data['open'].max()),
        'high': (data['high'].min(), data['high'].max()),
        'low': (data['low'].min(), data['low'].max()),
        'close': (data['close'].min(), data['close'].max())
    }
    stats['volume_stats'] = {
        'min': data['volume'].min(),
        'max': data['volume'].max(),
        'mean': data['volume'].mean()
    }
    stats['total_candles'] = len(data)
    return stats


# helper.py
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def plot(scores, mean_scores, portfolio_values):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.clear()
    ax1.plot(scores, label='Scores')
    ax1.plot(mean_scores, label='Mean Scores')
    ax1.set_title('Training Progress - Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax2.clear()
    ax2.plot(portfolio_values, label='Portfolio Value', color='green')
    ax2.set_title('Portfolio Value Over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Value ($)')
    ax2.legend()
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)
    plt.pause(0.1)

def plot_trade_analysis(trades_history):
    if not trades_history:
        print("No trades to analyze.")
        return
    trades_df = pd.DataFrame(trades_history)
    trades_df['pnl_sign'] = trades_df['pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss')
    win_loss_counts = trades_df['pnl_sign'].value_counts()
    plt.ioff()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    ax1.hist(trades_df['pnl'], bins=30, color='blue', alpha=0.7)
    ax1.set_title('Distribution of Trade P&L')
    ax1.set_xlabel('P&L')
    ax1.set_ylabel('Frequency')
    ax2.pie(win_loss_counts, labels=win_loss_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
    ax2.set_title('Win/Loss Ratio')
    cumulative_pnl = trades_df['pnl'].cumsum()
    ax3.plot(cumulative_pnl, marker='o', linestyle='-', color='purple')
    ax3.set_title('Cumulative P&L Over Trades')
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('Cumulative P&L')
    ax4.boxplot([trades_df[trades_df['pnl_sign'] == 'Win']['pnl'], trades_df[trades_df['pnl_sign'] == 'Loss']['pnl']], labels=['Wins', 'Losses'])
    ax4.set_title('P&L Box Plot (Wins vs Losses)')
    ax4.set_ylabel('P&L')
    plt.tight_layout()
    plt.show()


# trainer.py
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import os
from agent import Agent
from forex_game import ForexGameAI
from helper import plot, plot_trade_analysis

class V4Trainer:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.agent = Agent()
        self.game = None
        self.scores = []
        self.mean_scores = []
        self.portfolio_values = []
        self.total_score = 0
        self.record = 0
        self.best_portfolio = 1000

    def load_data(self):
        if self.data_path and os.path.exists(self.data_path):
            try:
                data = pd.read_csv(self.data_path)
                required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required_columns):
                    print(f"Data file missing required columns. Found: {data.columns.tolist()}")
                    return pd.DataFrame()
                data['time'] = pd.to_datetime(data['time'])
                data = data.sort_values('time').reset_index(drop=True)
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                data = data.dropna().reset_index(drop=True)
                return data
            except Exception as e:
                print(f"Error loading data: {e}")
                return pd.DataFrame()
        else:
            print("Data file path not provided or file does not exist.")
            return pd.DataFrame()

    def train(self, max_episodes=100, save_interval=10):
        print('=' * 60)
        print('V4 Forex DQN Training')
        print('=' * 60)
        print('Philosophy: Simplicity beats complexity')
        print('State: 4 simple inputs, Actions: 2 simple outputs')
        print('=' * 60)
        data = self.load_data()
        self.game = ForexGameAI(data)
        while self.agent.n_games < max_episodes:
            state_old = self.agent.get_state(self.game)
            final_move = self.agent.get_action(state_old)
            reward, done, portfolio_value = self.game.play_step(final_move.index(1))
            state_new = self.agent.get_state(self.game)
            self.agent.train_short_memory(state_old, final_move, reward, state_new, done)
            self.agent.remember(state_old, final_move, reward, state_new, done)
            if done:
                self.game.reset()
                self.agent.n_games += 1
                self.agent.train_long_memory()
                self.scores.append(reward)
                self.portfolio_values.append(portfolio_value)
                self.total_score += reward
                mean_score = np.mean(self.scores)
                self.mean_scores.append(mean_score)
                if portfolio_value > self.best_portfolio:
                    self.best_portfolio = portfolio_value
                    self.agent.save_model('best_model.pth')
                print(f'Episode {self.agent.n_games} | Portfolio: ${portfolio_value:.2f} | Reward: {reward:.2f}')
                if self.agent.n_games % save_interval == 0:
                    plot(self.scores, self.mean_scores, self.portfolio_values)
        self.save_final_results()

    def save_checkpoint(self):
        checkpoint = {
            'episode': self.agent.n_games,
            'model_state_dict': self.agent.model.state_dict(),
            'scores': self.scores,
            'mean_scores': self.mean_scores,
            'portfolio_values': self.portfolio_values,
            'record': self.record,
            'best_portfolio': self.best_portfolio
        }
        torch.save(checkpoint, f'checkpoint_episode_{self.agent.n_games}.pth')

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.agent.model.load_state_dict(checkpoint['model_state_dict'])
            self.agent.n_games = checkpoint['episode']
            self.scores = checkpoint['scores']
            self.mean_scores = checkpoint['mean_scores']
            self.portfolio_values = checkpoint['portfolio_values']
            self.record = checkpoint['record']
            self.best_portfolio = checkpoint['best_portfolio']
            print(f'Loaded checkpoint from episode {self.agent.n_games}')
            return True
        return False

    def save_final_results(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.agent.model.save(f'final_model_{timestamp}.pth')
        try:
            plot(self.scores, self.mean_scores, self.portfolio_values)
            plot_trade_analysis(self.game.trades_history)
        except Exception as e:
            print(f"Could not generate plots: {e}")