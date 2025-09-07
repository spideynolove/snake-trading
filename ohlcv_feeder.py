import pandas as pd
import numpy as np
from datetime import datetime
import time
import threading
import queue
from helpers import DataProcessor

class RealTimeOHLCVFeeder:
    def __init__(self, data_file, speed_multiplier=100, delimiter='\t', has_header=False, 
                 column_order=None, timestamp_format='%Y-%m-%d %H:%M'):
        self.data_file = data_file
        self.delimiter = delimiter
        self.has_header = has_header
        self.column_order = column_order or ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.timestamp_format = timestamp_format
        self.speed_multiplier = speed_multiplier
        
        self.raw_data = []
        self.current_index = -1
        self.running = False
        self.data_processor = DataProcessor()
        
        self.timestamps = []
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        self.volumes = []
        
        self.processed_df = pd.DataFrame()
        self.data_queue = queue.Queue()
        
        self._load_all_data()
    
    def _load_all_data(self):
        try:
            with open(self.data_file, 'r') as f:
                if self.has_header:
                    next(f)
                for line in f:
                    bar_data = self.parse_data_line(line)
                    if bar_data:
                        self.raw_data.append(bar_data)
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def parse_data_line(self, line):
        parts = line.strip().split(self.delimiter)
        if len(parts) != len(self.column_order):
            return None
        try:
            record = {}
            for i, col in enumerate(self.column_order):
                if col == 'timestamp':
                    record['timestamp'] = datetime.strptime(parts[i], self.timestamp_format)
                elif col == 'volume':
                    record['volume'] = int(parts[i])
                else:
                    record[col] = float(parts[i])
            return record
        except:
            return None
    
    def get_next_bar(self):
        if self.current_index + 1 >= len(self.raw_data):
            return None
        
        self.current_index += 1
        bar = self.raw_data[self.current_index]
        
        self.timestamps.append(bar['timestamp'])
        self.opens.append(bar['open'])
        self.highs.append(bar['high'])
        self.lows.append(bar['low'])
        self.closes.append(bar['close'])
        self.volumes.append(bar['volume'])
        
        self._update_processed_data()
        
        return bar
    
    def _update_processed_data(self):
        if len(self.closes) < 2:
            return
        
        current_df = pd.DataFrame({
            'timestamp': self.timestamps,
            'open': self.opens,
            'high': self.highs,
            'low': self.lows,
            'close': self.closes,
            'volume': self.volumes
        })
        
        self.processed_df = current_df.copy()
    
    def get_current_state(self):
        if self.current_index < 0:
            return None
        
        return {
            'current_bar': {
                'timestamp': self.timestamps[self.current_index],
                'open': self.opens[self.current_index],
                'high': self.highs[self.current_index],
                'low': self.lows[self.current_index],
                'close': self.closes[self.current_index],
                'volume': self.volumes[self.current_index]
            },
            'historical_data': {
                'timestamps': self.timestamps.copy(),
                'opens': self.opens.copy(),
                'highs': self.highs.copy(),
                'lows': self.lows.copy(),
                'closes': self.closes.copy(),
                'volumes': self.volumes.copy()
            },
            'processed_data': self.processed_df.copy() if not self.processed_df.empty else pd.DataFrame(),
            'current_index': self.current_index,
            'total_bars': len(self.raw_data)
        }
    
    def get_lookback_window(self, window_size=50):
        if self.current_index < 0:
            return None
        
        start_idx = max(0, self.current_index - window_size + 1)
        end_idx = self.current_index + 1
        
        return {
            'timestamps': self.timestamps[start_idx:end_idx],
            'opens': self.opens[start_idx:end_idx],
            'highs': self.highs[start_idx:end_idx],
            'lows': self.lows[start_idx:end_idx],
            'closes': self.closes[start_idx:end_idx],
            'volumes': self.volumes[start_idx:end_idx],
            'window_start_index': start_idx,
            'window_end_index': end_idx - 1
        }
    
    def has_next_bar(self):
        return self.current_index + 1 < len(self.raw_data)
    
    def reset(self):
        self.current_index = -1
        self.timestamps = []
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        self.volumes = []
        self.processed_df = pd.DataFrame()
    
    def simulate_trading_session(self, callback_func=None):
        self.running = True
        while self.running and self.has_next_bar():
            bar = self.get_next_bar()
            if bar is None:
                break
            
            if callback_func:
                callback_func(self.get_current_state())
            
            time.sleep(3600 / self.speed_multiplier / 1000)
        
        self.running = False
    
    def start_async_simulation(self, callback_func):
        def simulation_thread():
            self.simulate_trading_session(callback_func)
        
        thread = threading.Thread(target=simulation_thread, daemon=True)
        thread.start()
        return thread
    
    def stop(self):
        self.running = False
    
    def is_ready(self):
        return len(self.raw_data) > 0
    
    def get_total_bars(self):
        return len(self.raw_data)

if __name__ == "__main__":
    def trading_callback(state):
        current_bar = state['current_bar']
        print(f"Bar {state['current_index']}: {current_bar['timestamp']} "
              f"OHLC: {current_bar['open']:.5f}, {current_bar['high']:.5f}, "
              f"{current_bar['low']:.5f}, {current_bar['close']:.5f}")
        
        lookback = feeder.get_lookback_window(10)
        if lookback:
            recent_closes = lookback['closes']
            if len(recent_closes) >= 2:
                price_change = recent_closes[-1] - recent_closes[-2]
                print(f"  Price change: {price_change:.5f}")
    
    feeder = RealTimeOHLCVFeeder(
        data_file='data/XAUUSD60.csv',
        speed_multiplier=50,
        delimiter='\t',
        has_header=False,
        column_order=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        timestamp_format='%Y-%m-%d %H:%M'
    )
    
    print(f"Loaded {feeder.get_total_bars()} bars")
    
    try:
        feeder.simulate_trading_session(trading_callback)
    except KeyboardInterrupt:
        feeder.stop()
        print("Simulation stopped")