import pandas as pd
from core.agent import Agent
from env.forex_env import ForexEnv

def load_sample_data():
    # Create sample data for testing (you'll replace with real data loader)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    np_random = pd.Series(range(1000)).apply(lambda x: 1.2000 + (x % 100) * 0.0001)
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np_random,
        'high': np_random + 0.0005,
        'low': np_random - 0.0005,
        'close': np_random + pd.Series(range(1000)).apply(lambda x: (x % 20 - 10) * 0.0001),
        'volume': 1000
    })
    return data

def train():
    # Load data (replace with your real data loader)
    data = load_sample_data()
    
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
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        
        # Convert one-hot to action index
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
    train()