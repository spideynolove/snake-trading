import pandas as pd
import sys
from pathlib import Path
from core.agent import Agent
from env.forex_env import ForexEnv
from integration.data_feed import SequentialProcessor, DataFeedThread

def train(csv_path=None, use_sequential=True):
    if csv_path is None:
        print("Error: CSV file path required")
        print("Usage: python run.py --csv path/to/data.csv")
        return
    
    try:
        data_feed = DataFeedThread(csv_path=csv_path)
        data = data_feed.data
        print(f"Loaded {len(data)} data points from {csv_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = ForexEnv(data)
    
    print("Starting Snake-inspired Forex Trading AI training...")
    print(f"Data points: {len(data)}")
    print(f"Model: 4 inputs -> 256 hidden -> 3 outputs")
    print(f"Actions: 0=Close, 1=Long, 2=Short")
    print(f"Rewards: +10 profit, -10 loss")
    print(f"Processing mode: {'Sequential' if use_sequential else 'Threaded'}")
    
    if use_sequential:
        sequential_processor = SequentialProcessor(agent, game)
    
    while True:
        if use_sequential:
            try:
                final_balance = sequential_processor.process_episode(data)
                agent.n_games += 1
                agent.train_long_memory()
                
                if final_balance > record:
                    record = final_balance
                    agent.model.save()
                
                print(f'Game {agent.n_games}, Final Balance: ${final_balance:.2f}, Record: ${record:.2f}, Trades: {len(game.trades_history)}')
                
                plot_scores.append(final_balance)
                total_score += final_balance
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                
                game.reset()
                
            except Exception as e:
                print(f"Episode failed: {e}")
                break
        else:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            
            action = final_move.index(1)
            reward, done, score = game.step(action)
            state_new = agent.get_state(game)
            
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                
                final_balance = score
                if final_balance > record:
                    record = final_balance
                    agent.model.save()
                
                print(f'Game {agent.n_games}, Final Balance: ${final_balance:.2f}, Record: ${record:.2f}, Trades: {len(game.trades_history)}')
                
                plot_scores.append(final_balance)
                total_score += final_balance
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Snake-inspired Forex Trading AI')
    parser.add_argument('--csv', type=str, required=True, help='Path to GBPUSD H1 CSV file')
    parser.add_argument('--mode', choices=['sequential', 'threaded'], default='sequential', help='Processing mode')
    
    args = parser.parse_args()
    
    use_sequential = args.mode == 'sequential'
    train(csv_path=args.csv, use_sequential=use_sequential)