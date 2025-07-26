from core.agent import Agent
from env.forex_env import ForexEnv
from integration.data_feed import SequentialProcessor, BatchCSVProcessor, DataFeedThread

class OfflineTrainer:
    def __init__(self, csv_path, max_episodes=1000):
        self.csv_path = csv_path
        self.max_episodes = max_episodes
        self.agent = Agent()
        
        data_feed = DataFeedThread(csv_path=csv_path)
        self.data = data_feed.data
        self.env = ForexEnv(self.data)
        
    def train_single_episode(self):
        sequential_processor = SequentialProcessor(self.agent, self.env)
        final_balance = sequential_processor.process_episode(self.data)
        
        self.agent.n_games += 1
        self.agent.train_long_memory()
        
        return final_balance
    
    def train_batch_episodes(self, batch_size=1000):
        batch_processor = BatchCSVProcessor(self.csv_path, batch_size)
        total_episodes = batch_processor.process_all_batches(self.agent, self.env)
        
        self.agent.train_long_memory()
        return total_episodes
    
    def train(self, use_batches=False, batch_size=1000):
        best_score = 0
        scores = []
        
        for episode in range(self.max_episodes):
            if use_batches:
                total_episodes = self.train_batch_episodes(batch_size)
                final_balance = self.env.get_portfolio_value()
            else:
                final_balance = self.train_single_episode()
            
            scores.append(final_balance)
            
            if final_balance > best_score:
                best_score = final_balance
                self.agent.model.save(f'best_model_episode_{episode}.pth')
            
            self.env.reset()
            
            if episode % 10 == 0:
                avg_score = sum(scores[-10:]) / min(10, len(scores))
                print(f'Episode {episode}, Balance: ${final_balance:.2f}, Avg: ${avg_score:.2f}, Best: ${best_score:.2f}')
        
        return scores