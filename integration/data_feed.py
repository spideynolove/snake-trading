import threading
from queue import Queue, Empty
import pandas as pd
import time

class DataFeedThread(threading.Thread):
    def __init__(self, data, data_queue, synchronization_events, timeout=5.0):
        super().__init__()
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