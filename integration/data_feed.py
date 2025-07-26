import threading
from queue import Queue, Empty
import pandas as pd
import numpy as np
import time
from pathlib import Path

class DataFeedThread(threading.Thread):
    def __init__(self, data=None, csv_path=None, data_queue=None, synchronization_events=None, timeout=5.0):
        super().__init__()
        
        if csv_path is not None:
            self.data = self._load_csv_data(csv_path)
            self._validate_csv_data()
        else:
            self.data = data
            
        self.data_queue = data_queue
        self.sync_events = synchronization_events
        self.current_index = 0
        self.timeout = timeout
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        while self.current_index < len(self.data) and not self.stop_event.is_set():
            new_data_point = self.data.iloc[self.current_index]
            
            if self._is_market_gap(self.current_index):
                self.current_index += 1
                continue
            
            try:
                self.data_queue.put(new_data_point, timeout=self.timeout)
            except:
                break
            
            self.sync_events['F2'].set()
            
            if not self.sync_events['F1'].wait(timeout=self.timeout):
                break
            
            self.sync_events['F1'].clear()
            self.current_index += 1
    
    def _is_market_gap(self, index):
        if index == 0:
            return False
        
        current_time = pd.to_datetime(self.data.iloc[index]['timestamp'])
        prev_time = pd.to_datetime(self.data.iloc[index-1]['timestamp'])
        time_diff = (current_time - prev_time).total_seconds() / 3600
        
        return time_diff > 2
    
    def _load_csv_data(self, csv_path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        data = pd.read_csv(csv_path)
        
        if data.empty:
            raise ValueError("CSV file is empty")
        
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        except:
            raise ValueError("Cannot parse timestamp column")
        
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.dropna()
        
        if len(data) == 0:
            raise ValueError("No valid data after preprocessing")
        
        data = self._detect_market_gaps(data)
        data = self._normalize_features(data)
        
        return data
    
    def _detect_market_gaps(self, data):
        data = data.copy()
        data['time_diff'] = data['timestamp'].diff().dt.total_seconds() / 3600
        data['is_gap'] = data['time_diff'] > 2
        return data
    
    def _normalize_features(self, data):
        data = data.copy()
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                data[f'{col}_normalized'] = data[col] / data[col].iloc[0]
        
        if 'volume' in data.columns:
            data['volume_norm'] = (data['volume'] - data['volume'].min()) / (data['volume'].max() - data['volume'].min())
        
        return data
    
    def _validate_csv_data(self):
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        if self.data.empty:
            raise ValueError("CSV data is empty")
        
        if self.data['timestamp'].isnull().any():
            raise ValueError("CSV contains null timestamps")
    
    def stop(self):
        self.stop_event.set()

class TradingLogicThread(threading.Thread):
    def __init__(self, agent, env, data_queue, orders_queue, synchronization_events, timeout=5.0):
        super().__init__()
        self.agent = agent
        self.env = env
        self.data_queue = data_queue
        self.orders_queue = orders_queue
        self.sync_events = synchronization_events
        self.timeout = timeout
        self.daemon = True
        self.stop_event = threading.Event()
        self.performance_metrics = {'decision_times': []}

    def run(self):
        while not self.stop_event.is_set():
            if not self.sync_events['F2'].wait(timeout=self.timeout):
                continue
            
            try:
                current_data = self.data_queue.get(timeout=self.timeout)
            except Empty:
                continue
            
            start_time = time.time()
            
            self.env.temporal_constraints.set_current_data(current_data)
            
            state = self.agent.get_state(self.env)
            action = self.agent.get_action(state)
            
            decision_time = time.time() - start_time
            self.performance_metrics['decision_times'].append(decision_time)
            
            if any(action):
                action_index = action.index(1)
                try:
                    self.orders_queue.put({
                        'action': action_index,
                        'data': current_data,
                        'state': state,
                        'decision_time': decision_time
                    }, timeout=self.timeout)
                except:
                    pass
            
            self.sync_events['F3'].set()
            
            if not self.sync_events['F1'].wait(timeout=self.timeout):
                break
            
            self.sync_events['F2'].clear()
    
    def stop(self):
        self.stop_event.set()
    
    def get_performance_stats(self):
        if not self.performance_metrics['decision_times']:
            return {}
        
        times = self.performance_metrics['decision_times']
        return {
            'avg_decision_time': sum(times) / len(times),
            'max_decision_time': max(times),
            'decisions_count': len(times)
        }

class ExecutionThread(threading.Thread):
    def __init__(self, env, orders_queue, synchronization_events, timeout=5.0):
        super().__init__()
        self.env = env
        self.orders_queue = orders_queue
        self.sync_events = synchronization_events
        self.timeout = timeout
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            if not self.sync_events['F3'].wait(timeout=self.timeout):
                continue
            
            while not self.orders_queue.empty():
                try:
                    order = self.orders_queue.get(timeout=0.1)
                    reward, done, portfolio_value = self.env.step(order['action'])
                    
                    if done:
                        break
                except Empty:
                    break
                except Exception as e:
                    break
            
            self.sync_events['F1'].set()
            self.sync_events['F3'].clear()
    
    def stop(self):
        self.stop_event.set()

class SequentialProcessor:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
    
    def process_episode(self, data):
        for i, row in data.iterrows():
            self.env.temporal_constraints.set_current_data(row)
            self.env.current_step = i
            
            state = self.agent.get_state(self.env)
            action = self.agent.get_action(state)
            
            if any(action):
                action_index = action.index(1)
                reward, done, portfolio_value = self.env.step(action_index)
                
                if done:
                    break
        
        return self.env.get_portfolio_value()

class BatchCSVProcessor:
    def __init__(self, csv_path, batch_size=1000):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.data = self._load_csv_data(csv_path)
    
    def _load_csv_data(self, csv_path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        data = pd.read_csv(csv_path)
        
        if data.empty:
            raise ValueError("CSV file is empty")
        
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        except:
            raise ValueError("Cannot parse timestamp column")
        
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.dropna()
        
        if len(data) == 0:
            raise ValueError("No valid data after preprocessing")
        
        data = self._detect_market_gaps(data)
        data = self._normalize_features(data)
        
        return data
    
    def _detect_market_gaps(self, data):
        data = data.copy()
        data['time_diff'] = data['timestamp'].diff().dt.total_seconds() / 3600
        data['is_gap'] = data['time_diff'] > 2
        return data
    
    def _normalize_features(self, data):
        data = data.copy()
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                data[f'{col}_normalized'] = data[col] / data[col].iloc[0]
        
        if 'volume' in data.columns:
            data['volume_norm'] = (data['volume'] - data['volume'].min()) / (data['volume'].max() - data['volume'].min())
        
        return data
        
    def get_batches(self):
        for i in range(0, len(self.data), self.batch_size):
            yield self.data.iloc[i:i+self.batch_size]
    
    def process_all_batches(self, agent, env):
        total_episodes = 0
        for batch in self.get_batches():
            processor = SequentialProcessor(agent, env)
            processor.process_episode(batch)
            total_episodes += 1
            env.reset()
        
        return total_episodes