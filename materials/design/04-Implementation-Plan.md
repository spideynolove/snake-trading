# Final Trading System Implementation Plan
**Local Intelligence + Remote Execution Architecture**

## Overview
This plan creates a distributed Deep Q-Learning trading system that separates intelligence (local machine) from execution (VPS), using:
- **Local Machine**: Data crawling, training, inference, signal generation
- **VPS**: Signal reception, order execution via DWXConnect
- **Communication**: Encrypted Telegram bot for signal transmission

## System Architecture

### Local Machine (All Processing)
```
trading_system/
├── data_crawling/          # Real-time external data feeds
│   ├── market_data_feeds.py
│   ├── economic_calendar.py
│   └── intermarket_data.py
├── training/               # Gym-trading-env + DQN training
│   ├── train_dqn.py
│   └── backtest_validator.py
├── inference/              # Model predictions + signal generation
│   ├── signal_generator.py
│   └── risk_manager.py
└── communication/          # Signal transmission to VPS
    └── telegram_sender.py
```

### VPS (Execution Only)
```
execution_server/
├── signal_receiver/        # Telegram bot listener
│   └── telegram_receiver.py
├── dwxconnect/            # MT4 interface
│   └── (existing DWXConnect files)
└── order_executor/        # Execute received commands
    └── execution_engine.py
```

## Data Flow
```
External Data → DQN Model → Trading Signal → Telegram → VPS → DWXConnect → MT4
Local Machine   Local GPU    JSON Format    Bot API     Execution   Order API    Broker
```

## Signal Format (DWXConnect Compatible)
```json
{
  "symbol": "EURUSD",
  "order_type": "buy",       # buy/sell/close
  "lots": 0.01,
  "price": 0,                # 0 = market order
  "stop_loss": 0,
  "take_profit": 0,
  "magic": 12345,
  "comment": "DQN_Signal",
  "timestamp": "2024-01-15T10:30:00Z",
  "risk_percent": 1.0
}
```

## Component 1: External Data Crawling Pipeline

### File: `data_crawling/market_data_feeds.py`
```python
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class MarketDataCrawler:
    def __init__(self, symbols=['EURUSD', 'GBPUSD'], timeframes=['H1', 'H4', 'D1']):
        self.symbols = symbols
        self.timeframes = timeframes
        self.data_cache = {}
        self.last_update = {}
        
    async def fetch_oanda_data(self, symbol, timeframe, count=100):
        """Fetch live data from OANDA API (replace with your broker's API)"""
        # OANDA API example - replace with your data source
        headers = {
            'Authorization': 'Bearer YOUR_OANDA_TOKEN',
            'Content-Type': 'application/json'
        }
        
        url = f"https://api-fxtrade.oanda.com/v3/instruments/{symbol}/candles"
        params = {
            'granularity': timeframe,
            'count': count
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_oanda_response(data, symbol, timeframe)
                else:
                    print(f"Error fetching {symbol} {timeframe}: {response.status}")
                    return None
    
    def _process_oanda_response(self, data, symbol, timeframe):
        """Convert OANDA response to standard format"""
        candles = []
        for candle in data['candles']:
            if candle['complete']:
                candles.append({
                    'time': candle['time'],
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': candle['volume']
                })
        
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        return df
    
    async def update_all_data(self):
        """Update all symbol/timeframe combinations"""
        tasks = []
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                task = self.fetch_oanda_data(symbol, timeframe)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Store results
        idx = 0
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                if not isinstance(results[idx], Exception) and results[idx] is not None:
                    self.data_cache[f"{symbol}_{timeframe}"] = results[idx]
                    self.last_update[f"{symbol}_{timeframe}"] = datetime.now()
                idx += 1
    
    def get_features_for_symbol(self, symbol, primary_timeframe='H1'):
        """Extract features for DQN model"""
        key = f"{symbol}_{primary_timeframe}"
        if key not in self.data_cache:
            return None
            
        df = self.data_cache[key].copy()
        
        # Create features matching training format
        df['feature_close'] = df['close'].pct_change()
        df['feature_sma_5'] = df['close'].rolling(5).mean() / df['close']
        df['feature_sma_20'] = df['close'].rolling(20).mean() / df['close']
        df['feature_rsi'] = self._calculate_rsi(df['close'])
        df['feature_volume'] = df['volume'] / df['volume'].rolling(20).max()
        
        # Add multi-timeframe features
        h4_key = f"{symbol}_H4"
        if h4_key in self.data_cache:
            h4_trend = self._get_trend_direction(self.data_cache[h4_key])
            df['feature_h4_trend'] = h4_trend
        
        df.dropna(inplace=True)
        
        if len(df) > 0:
            return df.iloc[-1][['feature_close', 'feature_sma_5', 'feature_sma_20', 
                              'feature_rsi', 'feature_volume']].values
        return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100  # Normalize to 0-1
    
    def _get_trend_direction(self, df):
        """Simple trend direction: 1 if uptrend, -1 if downtrend, 0 if sideways"""
        if len(df) < 20:
            return 0
        
        sma_short = df['close'].rolling(5).mean().iloc[-1]
        sma_long = df['close'].rolling(20).mean().iloc[-1]
        
        if sma_short > sma_long * 1.001:  # 0.1% threshold
            return 1
        elif sma_short < sma_long * 0.999:
            return -1
        else:
            return 0
    
    async def start_real_time_feed(self, update_interval=60):
        """Start continuous data updates"""
        print("Starting real-time data feed...")
        while True:
            try:
                await self.update_all_data()
                print(f"Data updated at {datetime.now()}")
                await asyncio.sleep(update_interval)
            except Exception as e:
                print(f"Data update error: {e}")
                await asyncio.sleep(30)

# Usage
async def main():
    crawler = MarketDataCrawler(['EURUSD', 'GBPUSD'])
    await crawler.start_real_time_feed()

if __name__ == "__main__":
    asyncio.run(main())
```

### File: `data_crawling/economic_calendar.py`
```python
import requests
import pandas as pd
from datetime import datetime, timedelta
import json

class EconomicCalendar:
    def __init__(self):
        self.base_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.high_impact_events = []
        
    def fetch_calendar_data(self):
        """Fetch economic calendar from ForexFactory"""
        try:
            response = requests.get(self.base_url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching calendar: {response.status_code}")
                return None
        except Exception as e:
            print(f"Calendar fetch error: {e}")
            return None
    
    def filter_high_impact_events(self, data, currencies=['USD', 'EUR', 'GBP']):
        """Filter for high impact events affecting major currencies"""
        if not data:
            return []
            
        high_impact = []
        current_time = datetime.now()
        
        for event in data:
            try:
                event_time = datetime.fromisoformat(event.get('date', ''))
                
                # Only future events within next 24 hours
                if event_time > current_time and event_time < current_time + timedelta(hours=24):
                    
                    # High impact events only
                    if event.get('impact', '') == 'High':
                        
                        # Check if affects our currencies
                        currency = event.get('country', '').upper()
                        if any(curr in currency for curr in currencies):
                            high_impact.append({
                                'time': event_time,
                                'currency': currency,
                                'event': event.get('title', ''),
                                'impact': event.get('impact', ''),
                                'forecast': event.get('forecast', ''),
                                'previous': event.get('previous', '')
                            })
            except:
                continue
                
        return sorted(high_impact, key=lambda x: x['time'])
    
    def get_market_sentiment(self):
        """Get overall market sentiment based on upcoming events"""
        data = self.fetch_calendar_data()
        events = self.filter_high_impact_events(data)
        
        # Simple sentiment scoring
        if len(events) > 3:
            return "high_volatility"
        elif len(events) > 1:
            return "medium_volatility"
        else:
            return "low_volatility"
```

## Component 2: Signal Generation & Communication (Local Machine)

### File: `inference/signal_generator.py`
```python
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
import asyncio
import sys
sys.path.append('../data_crawling')
from market_data_feeds import MarketDataCrawler
from economic_calendar import EconomicCalendar

class SignalGenerator:
    def __init__(self, model_path, symbols=['EURUSD', 'GBPUSD']):
        self.symbols = symbols
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Data sources
        self.data_crawler = MarketDataCrawler(symbols)
        self.economic_cal = EconomicCalendar()
        
        # Risk management
        self.max_risk_per_trade = 0.02  # 2% risk per trade
        self.max_daily_trades = 5
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Position tracking
        self.current_positions = {symbol: 0 for symbol in symbols}  # -1, 0, 1
        
    def _load_model(self, model_path):
        """Load trained DQN model"""
        model = nn.Sequential(
            nn.Linear(5, 64),  # 5 features: close_change, sma_5, sma_20, rsi, volume
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)   # 3 actions: short, hold, long
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model
    
    def _reset_daily_counters(self):
        """Reset daily trade counters"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
    
    async def generate_signals(self):
        """Main signal generation loop"""
        print("Starting signal generation...")
        
        while True:
            try:
                self._reset_daily_counters()
                
                # Update market data
                await self.data_crawler.update_all_data()
                
                # Check economic calendar
                market_sentiment = self.economic_cal.get_market_sentiment()
                
                # Generate signals for each symbol
                for symbol in self.symbols:
                    signal = await self._analyze_symbol(symbol, market_sentiment)
                    if signal:
                        print(f"Generated signal: {signal}")
                        # Send to Telegram (implemented below)
                        await self._send_signal(signal)
                
                # Wait 5 minutes before next analysis
                await asyncio.sleep(300)
                
            except Exception as e:
                print(f"Signal generation error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_symbol(self, symbol, market_sentiment):
        """Analyze single symbol and generate signal if conditions met"""
        
        # Skip if already hit daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            return None
        
        # Get features from market data
        features = self.data_crawler.get_features_for_symbol(symbol)
        if features is None:
            return None
        
        # Get model prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            q_values = self.model(features_tensor)
            action = torch.argmax(q_values).item()
            confidence = torch.softmax(q_values, dim=1).max().item()
        
        # Convert action: 0=short, 1=hold, 2=long
        target_position = action - 1  # Convert to -1, 0, 1
        current_position = self.current_positions[symbol]
        
        # Only trade if:
        # 1. Position is changing
        # 2. Model confidence > 70%
        # 3. Market sentiment allows (avoid high volatility times)
        if (target_position != current_position and 
            confidence > 0.7 and 
            market_sentiment != "high_volatility"):
            
            # Calculate position size based on risk
            lot_size = self._calculate_position_size(symbol)
            
            # Create signal
            signal = {
                "symbol": symbol,
                "order_type": self._position_to_order_type(target_position, current_position),
                "lots": lot_size,
                "price": 0,  # Market order
                "stop_loss": 0,  # Calculated on VPS
                "take_profit": 0,  # Calculated on VPS
                "magic": 12345,
                "comment": f"DQN_Signal_conf_{confidence:.2f}",
                "timestamp": datetime.now().isoformat(),
                "risk_percent": self.max_risk_per_trade,
                "confidence": confidence,
                "market_sentiment": market_sentiment
            }
            
            # Update position tracking
            self.current_positions[symbol] = target_position
            self.daily_trade_count += 1
            
            return signal
        
        return None
    
    def _position_to_order_type(self, target_position, current_position):
        """Convert position change to order type"""
        if current_position != 0:
            # Close existing position first
            return "close"
        elif target_position == 1:
            return "buy"
        elif target_position == -1:
            return "sell"
        else:
            return "close"
    
    def _calculate_position_size(self, symbol):
        """Calculate position size based on risk management"""
        # Start with small size for testing
        base_size = 0.01
        
        # Adjust based on volatility (simplified)
        # In real implementation, use ATR or recent volatility
        return base_size
    
    async def _send_signal(self, signal):
        """Send signal via Telegram (placeholder - implement below)"""
        from telegram_sender import TelegramSender
        telegram = TelegramSender()
        await telegram.send_signal(signal)

# Usage
async def main():
    generator = SignalGenerator('models/gbpusd_dqn.pth', ['EURUSD', 'GBPUSD'])
    await generator.generate_signals()

if __name__ == "__main__":
    asyncio.run(main())
```

### File: `communication/telegram_sender.py`
```python
import asyncio
import json
import aiohttp
from datetime import datetime
import hashlib
import hmac

class TelegramSender:
    def __init__(self, bot_token, chat_id, encryption_key=None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.encryption_key = encryption_key
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    async def send_signal(self, signal_data):
        """Send trading signal via Telegram"""
        try:
            # Add authentication hash
            signal_data['auth_hash'] = self._generate_auth_hash(signal_data)
            
            # Convert to JSON
            message = json.dumps(signal_data, indent=2)
            
            # Send via Telegram
            await self._send_message(f"🤖 TRADING SIGNAL\n```json\n{message}\n```")
            
            print(f"Signal sent: {signal_data['symbol']} {signal_data['order_type']}")
            
        except Exception as e:
            print(f"Error sending signal: {e}")
    
    def _generate_auth_hash(self, signal_data):
        """Generate authentication hash for signal verification"""
        if not self.encryption_key:
            return "no_auth"
            
        # Create hash from critical signal data
        data_string = f"{signal_data['symbol']}{signal_data['order_type']}{signal_data['timestamp']}"
        return hmac.new(
            self.encryption_key.encode(),
            data_string.encode(),
            hashlib.sha256
        ).hexdigest()[:16]
    
    async def _send_message(self, text):
        """Send message via Telegram Bot API"""
        url = f"{self.base_url}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'Markdown'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status != 200:
                    print(f"Telegram send error: {response.status}")
                    
    async def send_status_update(self, message):
        """Send status/error messages"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await self._send_message(f"📊 STATUS [{timestamp}]\n{message}")
        
    async def send_performance_report(self, metrics):
        """Send daily performance report"""
        report = f"""
📈 DAILY PERFORMANCE REPORT
Date: {datetime.now().strftime("%Y-%m-%d")}

💰 P&L: {metrics.get('pnl', 'N/A')}
📊 Total Trades: {metrics.get('total_trades', 0)}
✅ Win Rate: {metrics.get('win_rate', 'N/A')}%
📉 Max Drawdown: {metrics.get('max_drawdown', 'N/A')}%
⏰ Active Since: {metrics.get('start_time', 'N/A')}
"""
        await self._send_message(report)

# Configuration
TELEGRAM_CONFIG = {
    'bot_token': 'YOUR_BOT_TOKEN',
    'chat_id': 'YOUR_CHAT_ID',
    'encryption_key': 'your_secret_key_for_auth'
}
```

## Component 3: VPS Execution Server

### File: `signal_receiver/telegram_receiver.py` (VPS)
```python
import asyncio
import json
import hashlib
import hmac
from datetime import datetime, timedelta
import aiohttp
import sys
sys.path.append('../order_executor')
from execution_engine import ExecutionEngine

class TelegramReceiver:
    def __init__(self, bot_token, encryption_key=None):
        self.bot_token = bot_token
        self.encryption_key = encryption_key
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.last_update_id = 0
        
        # Initialize execution engine
        self.executor = ExecutionEngine()
        
        # Signal validation
        self.processed_signals = set()  # Prevent duplicate processing
        
    async def start_listening(self):
        """Start listening for Telegram messages"""
        print("VPS: Starting Telegram signal listener...")
        
        while True:
            try:
                updates = await self._get_updates()
                
                for update in updates:
                    await self._process_update(update)
                
                await asyncio.sleep(2)  # Poll every 2 seconds
                
            except Exception as e:
                print(f"Telegram listener error: {e}")
                await asyncio.sleep(10)
    
    async def _get_updates(self):
        """Get new messages from Telegram"""
        url = f"{self.base_url}/getUpdates"
        params = {
            'offset': self.last_update_id + 1,
            'timeout': 30
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['ok'] and data['result']:
                        self.last_update_id = data['result'][-1]['update_id']
                        return data['result']
        
        return []
    
    async def _process_update(self, update):
        """Process incoming Telegram update"""
        if 'message' not in update:
            return
            
        message = update['message']
        text = message.get('text', '')
        
        # Look for trading signals (JSON format)
        if '🤖 TRADING SIGNAL' in text and '```json' in text:
            # Extract JSON from message
            try:
                json_start = text.find('```json') + 7
                json_end = text.find('```', json_start)
                json_text = text[json_start:json_end].strip()
                
                signal_data = json.loads(json_text)
                await self._process_signal(signal_data)
                
            except Exception as e:
                print(f"Error parsing signal: {e}")
    
    async def _process_signal(self, signal_data):
        """Process and validate trading signal"""
        try:
            # Validate signal structure
            required_fields = ['symbol', 'order_type', 'lots', 'timestamp', 'auth_hash']
            if not all(field in signal_data for field in required_fields):
                print("Invalid signal: missing required fields")
                return
            
            # Check signal age (reject if older than 5 minutes)
            signal_time = datetime.fromisoformat(signal_data['timestamp'])
            age = datetime.now() - signal_time
            if age > timedelta(minutes=5):
                print(f"Signal rejected: too old ({age.total_seconds():.0f}s)")
                return
            
            # Verify authentication
            if not self._verify_signal_auth(signal_data):
                print("Signal rejected: authentication failed")
                return
            
            # Check for duplicates
            signal_id = f"{signal_data['symbol']}_{signal_data['timestamp']}"
            if signal_id in self.processed_signals:
                print("Signal rejected: duplicate")
                return
            
            # Execute signal
            await self.executor.execute_signal(signal_data)
            
            # Mark as processed
            self.processed_signals.add(signal_id)
            
            # Cleanup old processed signals (keep last 100)
            if len(self.processed_signals) > 100:
                self.processed_signals = set(list(self.processed_signals)[-50:])
                
        except Exception as e:
            print(f"Signal processing error: {e}")
    
    def _verify_signal_auth(self, signal_data):
        """Verify signal authentication hash"""
        if not self.encryption_key or signal_data.get('auth_hash') == 'no_auth':
            return True  # No auth configured
            
        # Recreate hash
        data_string = f"{signal_data['symbol']}{signal_data['order_type']}{signal_data['timestamp']}"
        expected_hash = hmac.new(
            self.encryption_key.encode(),
            data_string.encode(),
            hashlib.sha256
        ).hexdigest()[:16]
        
        return expected_hash == signal_data.get('auth_hash')

# Usage
async def main():
    receiver = TelegramReceiver(
        bot_token='YOUR_BOT_TOKEN',
        encryption_key='your_secret_key_for_auth'
    )
    await receiver.start_listening()

if __name__ == "__main__":
    asyncio.run(main())
```

### File: `order_executor/execution_engine.py` (VPS)
```python
import sys
import os
sys.path.append('../dwxconnect/python')
from api.dwx_client import dwx_client
import time
import json
from datetime import datetime

class ExecutionEngine:
    def __init__(self, mt4_files_path):
        self.mt4_files_path = mt4_files_path
        self.dwx = dwx_client(metatrader_dir_path=mt4_files_path)
        
        # Risk limits (VPS-side safety)
        self.max_lot_size = 0.1
        self.max_positions_per_symbol = 1
        self.allowed_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        # Execution tracking
        self.execution_log = []
        
    async def execute_signal(self, signal_data):
        """Execute trading signal via DWXConnect"""
        try:
            symbol = signal_data['symbol']
            order_type = signal_data['order_type']
            lots = min(signal_data['lots'], self.max_lot_size)  # Safety limit
            
            # Validate symbol
            if symbol not in self.allowed_symbols:
                print(f"Signal rejected: {symbol} not in allowed symbols")
                return
            
            # Check existing positions
            open_positions = len([o for o in self.dwx.open_orders.values() 
                                if o.get('symbol') == symbol])
            
            if open_positions >= self.max_positions_per_symbol and order_type != 'close':
                print(f"Signal rejected: max positions reached for {symbol}")
                return
            
            # Execute order
            if order_type == 'buy':
                await self._execute_buy(symbol, lots, signal_data)
            elif order_type == 'sell':
                await self._execute_sell(symbol, lots, signal_data)
            elif order_type == 'close':
                await self._execute_close(symbol, signal_data)
            
        except Exception as e:
            print(f"Execution error: {e}")
            self._log_execution(signal_data, success=False, error=str(e))
    
    async def _execute_buy(self, symbol, lots, signal_data):
        """Execute buy order"""
        try:
            # Get current market price
            if symbol in self.dwx.market_data:
                ask_price = self.dwx.market_data[symbol]['ask']
                
                # Calculate stop loss and take profit
                sl_price = self._calculate_stop_loss(ask_price, 'buy', signal_data)
                tp_price = self._calculate_take_profit(ask_price, 'buy', signal_data)
                
                # Place order
                self.dwx.open_order(
                    symbol=symbol,
                    order_type='buy',
                    price=ask_price,
                    lots=lots,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    comment=signal_data.get('comment', 'DQN_Signal')
                )
                
                print(f"BUY order placed: {symbol} {lots} lots at {ask_price}")
                self._log_execution(signal_data, success=True)
                
            else:
                print(f"No market data for {symbol}")
                
        except Exception as e:
            print(f"Buy execution error: {e}")
            self._log_execution(signal_data, success=False, error=str(e))
    
    async def _execute_sell(self, symbol, lots, signal_data):
        """Execute sell order"""
        try:
            if symbol in self.dwx.market_data:
                bid_price = self.dwx.market_data[symbol]['bid']
                
                sl_price = self._calculate_stop_loss(bid_price, 'sell', signal_data)
                tp_price = self._calculate_take_profit(bid_price, 'sell', signal_data)
                
                self.dwx.open_order(
                    symbol=symbol,
                    order_type='sell',
                    price=bid_price,
                    lots=lots,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    comment=signal_data.get('comment', 'DQN_Signal')
                )
                
                print(f"SELL order placed: {symbol} {lots} lots at {bid_price}")
                self._log_execution(signal_data, success=True)
                
        except Exception as e:
            print(f"Sell execution error: {e}")
            self._log_execution(signal_data, success=False, error=str(e))
    
    async def _execute_close(self, symbol, signal_data):
        """Close all positions for symbol"""
        try:
            closed_count = 0
            for order_id, order_info in list(self.dwx.open_orders.items()):
                if order_info.get('symbol') == symbol:
                    self.dwx.close_order(order_id)
                    closed_count += 1
            
            print(f"Closed {closed_count} positions for {symbol}")
            self._log_execution(signal_data, success=True)
            
        except Exception as e:
            print(f"Close execution error: {e}")
            self._log_execution(signal_data, success=False, error=str(e))
    
    def _calculate_stop_loss(self, entry_price, direction, signal_data):
        """Calculate stop loss price"""
        risk_pips = 20  # Default 20 pip stop loss
        
        if direction == 'buy':
            return entry_price - (risk_pips * 0.0001)  # For EURUSD/GBPUSD
        else:
            return entry_price + (risk_pips * 0.0001)
    
    def _calculate_take_profit(self, entry_price, direction, signal_data):
        """Calculate take profit price"""
        profit_pips = 40  # Default 2:1 risk/reward
        
        if direction == 'buy':
            return entry_price + (profit_pips * 0.0001)
        else:
            return entry_price - (profit_pips * 0.0001)
    
    def _log_execution(self, signal_data, success=True, error=None):
        """Log execution results"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal_data,
            'success': success,
            'error': error
        }
        
        self.execution_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.execution_log) > 1000:
            self.execution_log = self.execution_log[-500:]
    
    def get_execution_stats(self):
        """Get execution statistics"""
        if not self.execution_log:
            return {}
        
        total_executions = len(self.execution_log)
        successful_executions = sum(1 for log in self.execution_log if log['success'])
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': (successful_executions / total_executions) * 100,
            'last_execution': self.execution_log[-1]['timestamp'] if self.execution_log else None
        }

# Usage
def main():
    MT4_FILES_PATH = "/path/to/mt4/files"  # Configure for your VPS
    engine = ExecutionEngine(MT4_FILES_PATH)
    
    # Test signal
    test_signal = {
        "symbol": "EURUSD",
        "order_type": "buy",
        "lots": 0.01,
        "timestamp": datetime.now().isoformat(),
        "comment": "Test_Signal"
    }
    
    import asyncio
    asyncio.run(engine.execute_signal(test_signal))

if __name__ == "__main__":
    main()
```

## Component 4: Risk Management & Monitoring

### File: `inference/risk_manager.py` (Local Machine)
```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class RiskManager:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = 0.02  # 2%
        self.max_daily_loss = 0.05  # 5%
        self.max_drawdown = 0.10  # 10%
        
        # Position tracking
        self.open_positions = {}
        self.daily_pnl = 0
        self.peak_balance = initial_balance
        self.trade_history = []
        
        # Daily limits
        self.max_trades_per_day = 5
        self.daily_trade_count = 0
        self.last_trade_date = None
        
    def validate_signal(self, signal_data):
        """Validate signal against risk parameters"""
        
        # Reset daily counters if new day
        self._reset_daily_counters()
        
        # Check daily trade limit
        if self.daily_trade_count >= self.max_trades_per_day:
            return False, "Daily trade limit reached"
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.initial_balance
        if self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss:
            return False, f"Daily loss limit reached: {daily_loss_pct:.2%}"
        
        # Check max drawdown
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown >= self.max_drawdown:
            return False, f"Max drawdown reached: {current_drawdown:.2%}"
        
        # Validate position size
        if signal_data['lots'] > self._calculate_max_position_size():
            return False, "Position size exceeds risk limits"
        
        return True, "Signal approved"
    
    def _reset_daily_counters(self):
        """Reset daily counters at start of new day"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.daily_pnl = 0
            self.last_trade_date = today
    
    def _calculate_max_position_size(self):
        """Calculate maximum position size based on current balance"""
        max_risk_amount = self.current_balance * self.max_risk_per_trade
        # Assuming 20 pip stop loss and EURUSD
        pip_value = 1  # $1 per pip for 0.01 lot EURUSD
        stop_loss_pips = 20
        max_lots = max_risk_amount / (stop_loss_pips * pip_value)
        return min(max_lots, 0.1)  # Cap at 0.1 lots
    
    def update_position(self, symbol, action, lots, price):
        """Update position tracking"""
        self.daily_trade_count += 1
        
        position = {
            'symbol': symbol,
            'action': action,
            'lots': lots,
            'entry_price': price,
            'timestamp': datetime.now(),
            'pnl': 0
        }
        
        self.open_positions[symbol] = position
        
    def calculate_unrealized_pnl(self, symbol, current_price):
        """Calculate unrealized P&L for open position"""
        if symbol not in self.open_positions:
            return 0
        
        position = self.open_positions[symbol]
        if position['action'] == 'buy':
            pnl = (current_price - position['entry_price']) * position['lots'] * 100000
        else:  # sell
            pnl = (position['entry_price'] - current_price) * position['lots'] * 100000
        
        return pnl
    
    def close_position(self, symbol, exit_price):
        """Close position and update balance"""
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions[symbol]
        realized_pnl = self.calculate_unrealized_pnl(symbol, exit_price)
        
        # Update balances
        self.current_balance += realized_pnl
        self.daily_pnl += realized_pnl
        self.peak_balance = max(self.peak_balance, self.current_balance)
        
        # Log trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': position['action'],
            'lots': position['lots'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl': realized_pnl,
            'balance_after': self.current_balance
        }
        
        self.trade_history.append(trade_record)
        del self.open_positions[symbol]
        
        return realized_pnl
    
    def get_risk_metrics(self):
        """Calculate current risk metrics"""
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        
        # Calculate win rate from recent trades
        recent_trades = self.trade_history[-20:] if len(self.trade_history) >= 20 else self.trade_history
        winning_trades = sum(1 for trade in recent_trades if trade['pnl'] > 0)
        win_rate = (winning_trades / len(recent_trades)) * 100 if recent_trades else 0
        
        return {
            'current_balance': self.current_balance,
            'total_return_pct': total_return * 100,
            'current_drawdown_pct': current_drawdown * 100,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trade_count,
            'open_positions': len(self.open_positions),
            'win_rate': win_rate,
            'total_trades': len(self.trade_history)
        }
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        print("⚠️ EMERGENCY STOP TRIGGERED")
        print(f"Current Drawdown: {((self.peak_balance - self.current_balance) / self.peak_balance) * 100:.2f}%")
        print(f"Daily P&L: ${self.daily_pnl:.2f}")
        
        return {
            'action': 'emergency_stop',
            'reason': 'Risk limits exceeded',
            'metrics': self.get_risk_metrics()
        }
```

## Implementation Timeline & Steps

### Phase 1: Local Development (Week 1-2)
**Prerequisites:**
```bash
# Local machine setup
pip install torch pandas numpy aiohttp requests gym-trading-env
pip install python-telegram-bot asyncio
```

**Day 1-3: Training Pipeline**
1. Download your GBPUSD data to `/data/GBPUSD60.csv`
2. Run training script with gym-trading-env
3. Validate backtest results
4. Save trained model

**Day 4-7: Signal Generation**
1. Implement data crawling (start with historical data)
2. Create signal generator with trained model
3. Test signal generation without execution
4. Integrate risk management validation

### Phase 2: Communication Setup (Week 2)
**Prerequisites:**
```bash
# Create Telegram bot
# 1. Message @BotFather on Telegram
# 2. Create new bot: /newbot
# 3. Get bot token and chat ID
```

**Day 8-10: Local Signal Sender**
1. Configure Telegram bot credentials
2. Test signal sending from local machine
3. Implement signal authentication/encryption
4. Test end-to-end signal flow

**Day 11-14: VPS Signal Receiver**
1. Set up minimal VPS (Ubuntu 20.04+)
2. Install Python 3.8+ and basic packages only
3. Deploy Telegram receiver
4. Test signal reception and parsing

### Phase 3: VPS Execution (Week 3)
**Prerequisites:**
```bash
# VPS setup (minimal)
sudo apt update && sudo apt install python3 python3-pip
pip3 install aiohttp requests
```

**Day 15-17: DWXConnect Integration**
1. Install MT4/MT5 on VPS
2. Deploy DWXConnect EA
3. Test basic order execution
4. Configure execution engine

**Day 18-21: End-to-End Testing**
1. Paper trading with demo account
2. Signal latency testing
3. Risk management validation
4. Error handling and recovery

### Phase 4: Live Deployment (Week 4+)
**Day 22-28: Production Deployment**
1. Switch to live broker account
2. Start with 0.01 lot sizes
3. Monitor performance daily
4. Gradual position size increase

## Risk Controls Summary

### Local Machine (Signal Generation)
- **Model Confidence**: Only trade signals > 70% confidence
- **Daily Trade Limit**: Maximum 5 trades per day
- **Economic Calendar**: Avoid high volatility periods
- **Position Size**: 2% risk per trade maximum
- **Market Hours**: Only trade during active sessions

### Communication (Telegram)
- **Signal Authentication**: HMAC-SHA256 signature verification
- **Age Validation**: Reject signals older than 5 minutes
- **Duplicate Prevention**: Track processed signals
- **Encryption**: Optional signal encryption for security

### VPS (Execution)
- **Symbol Whitelist**: Only execute approved symbols
- **Position Limits**: Maximum 1 position per symbol
- **Lot Size Cap**: 0.1 lots maximum per trade
- **Connection Monitoring**: Auto-restart on MT4 disconnection
- **Emergency Stop**: Manual override capability

### Overall System
- **Drawdown Limit**: Stop trading at 10% drawdown
- **Daily Loss Limit**: Stop trading at 5% daily loss
- **Win Rate Monitoring**: Alert if win rate drops below 30%
- **Performance Reporting**: Daily Telegram reports

## Testing Checklist

### Unit Tests
- [ ] Signal generation with various market conditions
- [ ] Risk management validation logic
- [ ] Telegram authentication/encryption
- [ ] DWXConnect order execution
- [ ] Position size calculations

### Integration Tests  
- [ ] End-to-end signal flow (Local → Telegram → VPS → MT4)
- [ ] Signal latency measurement
- [ ] Error handling and recovery
- [ ] Network disconnection scenarios
- [ ] MT4 connection failures

### Paper Trading
- [ ] 2 weeks minimum demo account testing
- [ ] Signal accuracy tracking
- [ ] Performance metrics validation
- [ ] Risk management effectiveness
- [ ] System stability monitoring

## Success Metrics

### Training Phase
- **Backtest Sharpe Ratio**: > 0.5 on out-of-sample data
- **Max Drawdown**: < 15% during training validation
- **Win Rate**: > 45% on historical data

### Live Trading Phase
- **Signal Execution**: > 95% successful execution rate
- **Latency**: < 10 seconds from signal to execution
- **Uptime**: > 99% system availability
- **Risk Compliance**: Zero risk limit violations

### Performance Targets (Month 1)
- **Capital Preservation**: No losses > 5% in single day
- **Consistency**: Positive performance in 60%+ of trading days
- **Risk-Adjusted Returns**: Sharpe ratio > 0.3
- **System Reliability**: < 1% downtime

This architecture separates intelligence from execution, keeps the VPS minimal and reliable, uses simple Telegram communication, and includes comprehensive risk management at every level.

---

## Data Crawling Desired Output

### Context
The user has an existing sophisticated trading strategy in `/plan/` that includes:
- **Fibonacci Extension Strategy** with supply/demand zones (1.78-1.88, 2.78-2.88 zones)
- **Multi-timeframe analysis** (H1, H4, D1, W1)
- **Economic calendar integration** with high-impact event filtering
- **Intermarket correlation analysis** (DXY, Gold, S&P500, 10Y Treasury, Oil)
- **TA Indicator Pipeline** with grid search optimization
- **DQN Implementation Plan** that enhances rule-based system

The data crawling component must support this existing strategy, NOT replace it with simple features.

### 1. Multi-Timeframe Market Data

**Purpose**: Support Fibonacci extension analysis across multiple timeframes and swing point identification.

**Fields per record**:
```python
{
    'time': datetime,           # Candle timestamp (timezone-aware UTC)
    'open': float,              # Opening price (5+ decimal precision)
    'high': float,              # Highest price (5+ decimal precision)
    'low': float,               # Lowest price (5+ decimal precision)
    'close': float,             # Closing price (5+ decimal precision)
    'volume': int,              # Trading volume
    'symbol': str,              # 'EURUSD', 'GBPUSD', 'USDJPY', etc.
    'timeframe': str            # 'H1', 'H4', 'D1', 'W1'
}
```

**Required Coverage**:
- **Symbols**: At minimum EURUSD, GBPUSD (expandable to other majors)
- **Timeframes**: H1, H4, D1, W1 (for multi-timeframe trend analysis)
- **History Depth**: Minimum 200 candles per timeframe (for swing analysis)
- **Update Frequency**: Real-time or 1-minute lag maximum

**Data Structure**:
```python
# Stored in MarketDataCrawler.data_cache
data_cache = {
    'EURUSD_H1': pandas.DataFrame,   # Last 200 H1 candles
    'EURUSD_H4': pandas.DataFrame,   # Last 200 H4 candles  
    'EURUSD_D1': pandas.DataFrame,   # Last 200 D1 candles
    'EURUSD_W1': pandas.DataFrame,   # Last 200 W1 candles
    'GBPUSD_H1': pandas.DataFrame,   # Last 200 H1 candles
    'GBPUSD_H4': pandas.DataFrame,   # Last 200 H4 candles
    'GBPUSD_D1': pandas.DataFrame,   # Last 200 D1 candles
    'GBPUSD_W1': pandas.DataFrame    # Last 200 W1 candles
}

# DataFrame index: DatetimeIndex (UTC)
# DataFrame columns: ['open', 'high', 'low', 'close', 'volume']
```

### 2. Economic Calendar Data

**Purpose**: Filter trading during high-impact news events and adjust position sizing based on volatility expectations.

**Fields per event**:
```python
{
    'time': datetime,           # Event scheduled time (UTC)
    'currency': str,            # 'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD'
    'event': str,               # 'Non-Farm Payrolls', 'CPI', 'FOMC Decision', etc.
    'impact': str,              # 'High', 'Medium', 'Low'
    'forecast': str,            # Expected value (if available)
    'previous': str,            # Previous release value (if available)
    'actual': str,              # Actual value (None if not released yet)
    'country': str,             # 'United States', 'United Kingdom', etc.
    'volatility_expected': str  # 'Low', 'Moderate', 'High' volatility expected
}
```

**Required Coverage**:
- **Lookforward**: Next 48 hours of events
- **Impact Filter**: High and Medium impact events only
- **Currency Filter**: Events affecting USD, EUR, GBP, JPY (major pairs)
- **Update Frequency**: Every 6 hours or when new events are added

**Data Structure**:
```python
# Returned by EconomicCalendar.filter_high_impact_events()
upcoming_events = [
    {
        'time': datetime(2024, 1, 15, 13, 30),  # UTC
        'currency': 'USD',
        'event': 'CPI m/m',
        'impact': 'High',
        'forecast': '0.3%',
        'previous': '0.1%',
        'actual': None,
        'country': 'United States',
        'volatility_expected': 'High'
    },
    # ... more events
]

# Market sentiment derived from events
market_sentiment = "high_volatility" | "medium_volatility" | "low_volatility"
```

### 3. Intermarket Correlation Data

**Purpose**: Support correlation-based trade filtering and confirm trend direction across multiple asset classes.

**Fields per asset**:
```python
{
    'time': datetime,           # Price timestamp (UTC)
    'symbol': str,              # Asset symbol
    'price': float,             # Current/close price
    'change_pct_daily': float,  # Daily percentage change
    'change_pct_weekly': float, # Weekly percentage change
    'volume': int               # Trading volume (if available)
}
```

**Required Assets** (matching `/data/intermarket-data/`):
- **DXY**: US Dollar Index
- **XAUUSD**: Gold spot price
- **SPX500**: S&P 500 index
- **US10Y**: 10-year Treasury yield
- **BRENTUSD**: Brent crude oil
- **WTIUSD**: WTI crude oil
- **Additional**: AUDUSD, EURGBP, GBPJPY, USDJPY, UK100, UKGILT10Y

**Data Structure**:
```python
# Stored in separate correlation data cache
correlation_data = {
    'DXY': {
        'time': datetime(2024, 1, 15, 14, 30),
        'price': 103.45,
        'change_pct_daily': -0.23,
        'change_pct_weekly': 1.15,
        'volume': 0
    },
    'XAUUSD': {
        'time': datetime(2024, 1, 15, 14, 30),
        'price': 2045.30,
        'change_pct_daily': 0.45,
        'change_pct_weekly': -1.20,
        'volume': 0
    },
    # ... other assets
}
```

### 4. Data Freshness Tracking

**Purpose**: Ensure signal generator only uses recent data and can detect stale data issues.

**Fields**:
```python
last_update_timestamps = {
    'EURUSD_H1': datetime(2024, 1, 15, 14, 30, 15),
    'EURUSD_H4': datetime(2024, 1, 15, 14, 00, 10),
    'EURUSD_D1': datetime(2024, 1, 15, 00, 00, 05),
    'EURUSD_W1': datetime(2024, 1, 14, 22, 00, 00),
    'economic_calendar': datetime(2024, 1, 15, 14, 25, 00),
    'correlation_data': datetime(2024, 1, 15, 14, 30, 00)
}
```

### 5. Interface Methods for Signal Generator

The signal generator will call these methods to get processed data:

```python
# Get raw OHLCV data for technical analysis
df_h1 = data_crawler.get_timeframe_data('EURUSD', 'H1')
df_h4 = data_crawler.get_timeframe_data('EURUSD', 'H4')
df_d1 = data_crawler.get_timeframe_data('EURUSD', 'D1')

# Get economic events for next 24 hours
events = economic_calendar.get_upcoming_events(hours_ahead=24)
market_sentiment = economic_calendar.get_market_sentiment()

# Get correlation data for intermarket analysis
corr_data = correlation_analyzer.get_current_correlations()

# Check data freshness (reject if older than thresholds)
data_age = data_crawler.get_data_age('EURUSD_H1')
is_fresh = data_age < timedelta(minutes=5)  # H1 data should be < 5 min old
```

### 6. Signal Generator Responsibilities

The signal generator (NOT data crawling) will:
- Apply the existing TA Indicator Pipeline from `/plan/04-ta-indicator-pipeline.md`
- Calculate Fibonacci extension zones (1.78-1.88, 2.78-2.88) from `/plan/05-fibonacci-extension-guide.md`
- Identify supply/demand zones using swing analysis
- Apply correlation filters based on DXY, Gold, etc.
- Generate enhanced state features for the DQN model from `/plan/10-dqn-implementation-plan.md`
- Execute the rule-based logic before DQN enhancement

### 7. Update Frequencies & Data Requirements

| Data Type | Update Frequency | Staleness Threshold | Critical for Strategy |
|-----------|------------------|---------------------|----------------------|
| H1 OHLCV | Every hour | 5 minutes | ✅ Swing analysis |
| H4 OHLCV | Every 4 hours | 30 minutes | ✅ Trend confirmation |
| D1 OHLCV | Daily | 2 hours | ✅ Major trend direction |
| W1 OHLCV | Weekly | 1 day | ✅ Long-term trend |
| Economic Events | 6 hours | 12 hours | ✅ Risk filtering |
| Correlation Data | 1 hour | 4 hours | ✅ Trade confirmation |

**Critical Success Criteria:**
- Data must support existing Fibonacci strategy (not replace it)
- Must provide multi-timeframe view for swing analysis
- Must include economic calendar for risk management
- Must support correlation-based trade filtering
- Raw data only - signal generator handles all technical analysis
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def prepare_data():
    # Load GBPUSD data
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Create features for gym-trading-env
    df['feature_close'] = df['close'].pct_change()
    df['feature_sma_5'] = df['close'].rolling(5).mean() / df['close']
    df['feature_sma_20'] = df['close'].rolling(20).mean() / df['close']
    df['feature_volume'] = df['volume'] / df['volume'].rolling(20).max()
    df.dropna(inplace=True)
    
    return df

def train_agent():
    # Prepare data
    df = prepare_data()
    
    # Create environment
    env = TradingEnv(
        df=df,
        positions=[-1, 0, 1],  # Short, Hold, Long
        reward_function=lambda hist: np.log(hist["portfolio_valuation", -1] / hist["portfolio_valuation", -2]),
        trading_fees=0.0001,  # 1 pip spread
        portfolio_initial_value=10000
    )
    
    # Initialize agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Training loop
    scores = []
    for episode in range(EPISODES):
        state = env.reset()[0]
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.remember(state, action, reward, next_state, done or truncated)
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
                
        agent.replay(BATCH_SIZE)
        
        if episode % 50 == 0:
            agent.update_target_network()
            
        scores.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    # Save model
    torch.save(agent.q_network.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    return agent, scores

if __name__ == "__main__":
    agent, scores = train_agent()
```

### Dependencies for Training
```bash
pip install gym-trading-env torch pandas numpy matplotlib
```

## Component 2: Bridge Module

### File: `trading_bridge.py`
```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import deque
import time

class TradingBridge:
    def __init__(self, model_path, dwx_client):
        self.dwx = dwx_client
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # State management
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.current_position = 0  # -1: short, 0: flat, 1: long
        self.last_action_time = 0
        self.min_action_interval = 60  # Minimum 60 seconds between actions
        
        # Feature calculation
        self.sma_5 = deque(maxlen=5)
        self.sma_20 = deque(maxlen=20)
        
    def _load_model(self, model_path):
        # Match the training model architecture
        model = nn.Sequential(
            nn.Linear(4, 64),  # 4 features: close_change, sma_5, sma_20, volume
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)   # 3 actions: short, hold, long
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model
    
    def process_tick(self, symbol, bid, ask, volume=0):
        """Main entry point called from DWX on_tick"""
        if symbol != 'GBPUSD':
            return
            
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_action_time < self.min_action_interval:
            return
            
        # Update price history
        mid_price = (bid + ask) / 2
        self.price_history.append(mid_price)
        self.volume_history.append(volume)
        
        # Need at least 20 prices for features
        if len(self.price_history) < 20:
            return
            
        # Calculate features
        state = self._calculate_state()
        if state is None:
            return
            
        # Get action from model
        action = self._predict_action(state)
        
        # Execute action
        self._execute_action(action, symbol, bid, ask)
        
        self.last_action_time = current_time
    
    def _calculate_state(self):
        """Convert market data to gym-compatible state"""
        try:
            prices = np.array(list(self.price_history))
            volumes = np.array(list(self.volume_history))
            
            # Feature 1: Price change (pct_change)
            if len(prices) < 2:
                return None
            price_change = (prices[-1] - prices[-2]) / prices[-2]
            
            # Feature 2: SMA 5 ratio
            sma_5 = np.mean(prices[-5:]) / prices[-1] if len(prices) >= 5 else 1.0
            
            # Feature 3: SMA 20 ratio  
            sma_20 = np.mean(prices[-20:]) / prices[-1] if len(prices) >= 20 else 1.0
            
            # Feature 4: Volume ratio
            volume_ratio = volumes[-1] / np.max(volumes[-20:]) if len(volumes) >= 20 and np.max(volumes[-20:]) > 0 else 0.0
            
            state = np.array([price_change, sma_5, sma_20, volume_ratio], dtype=np.float32)
            
            # Handle NaN/inf values
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state
            
        except Exception as e:
            print(f"Error calculating state: {e}")
            return None
    
    def _predict_action(self, state):
        """Get action from trained model"""
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                action = torch.argmax(q_values).item()
            return action
        except Exception as e:
            print(f"Error predicting action: {e}")
            return 1  # Default to hold
    
    def _execute_action(self, action, symbol, bid, ask):
        """Translate model action to DWX commands"""
        try:
            # Action mapping: 0=short, 1=hold, 2=long
            target_position = action - 1  # Convert to -1, 0, 1
            
            if target_position == self.current_position:
                return  # No change needed
                
            lots = 0.01  # Small position size for testing
            
            # Close existing position first if changing direction
            if self.current_position != 0:
                self.dwx.close_orders_by_symbol(symbol)
                self.current_position = 0
                print(f"Closed position for {symbol}")
                time.sleep(1)  # Brief pause
            
            # Open new position
            if target_position == 1:  # Go long
                self.dwx.open_order(symbol, 'buy', ask, lots)
                self.current_position = 1
                print(f"Opened LONG position: {symbol} at {ask}")
                
            elif target_position == -1:  # Go short
                self.dwx.open_order(symbol, 'sell', bid, lots)
                self.current_position = -1
                print(f"Opened SHORT position: {symbol} at {bid}")
                
            # target_position == 0 means stay flat (already handled above)
            
        except Exception as e:
            print(f"Error executing action: {e}")
    
    def get_status(self):
        """Return current status for monitoring"""
        return {
            'current_position': self.current_position,
            'price_history_length': len(self.price_history),
            'last_action_time': self.last_action_time
        }
```

## Component 3: Live Trading Interface (DWXConnect Integration)

### File: `live_trader.py`
```python
import sys
import os
sys.path.append('path/to/dwxconnect/python')  # Add DWX path

from api.dwx_client import dwx_client
from trading_bridge import TradingBridge
import time
import threading

class LiveTrader:
    def __init__(self, mt4_files_path, model_path):
        self.mt4_files_path = mt4_files_path
        self.model_path = model_path
        
        # Initialize DWX client
        self.dwx = dwx_client(metatrader_dir_path=mt4_files_path)
        
        # Initialize bridge
        self.bridge = TradingBridge(model_path, self.dwx)
        
        # Trading state
        self.is_trading = False
        self.trade_count = 0
        
    def start_trading(self):
        """Start live trading"""
        print("Starting live trading...")
        
        # Subscribe to GBPUSD
        self.dwx.subscribe_symbols(['GBPUSD'])
        
        # Set up tick processing
        self.dwx.tick_processor = self._process_tick
        
        self.is_trading = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("Live trading started. Press Ctrl+C to stop.")
        
        try:
            # Keep main thread alive
            while self.is_trading:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_trading()
    
    def _process_tick(self, symbol, bid, ask):
        """Called by DWX on each tick"""
        if symbol == 'GBPUSD':
            # Get volume from market data if available
            volume = 0
            if hasattr(self.dwx, 'market_data') and symbol in self.dwx.market_data:
                volume = self.dwx.market_data[symbol].get('volume', 0)
            
            # Process through bridge
            self.bridge.process_tick(symbol, bid, ask, volume)
    
    def _monitor_loop(self):
        """Monitor trading status"""
        while self.is_trading:
            try:
                # Print status every 5 minutes
                status = self.bridge.get_status()
                print(f"Status: Position={status['current_position']}, "
                      f"Price History={status['price_history_length']}")
                
                # Check for orders
                if hasattr(self.dwx, 'open_orders'):
                    print(f"Open orders: {len(self.dwx.open_orders)}")
                
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(60)
    
    def stop_trading(self):
        """Stop trading and cleanup"""
        print("Stopping trading...")
        self.is_trading = False
        
        # Close all positions
        try:
            self.dwx.close_orders_by_symbol('GBPUSD')
            print("Closed all GBPUSD positions")
        except:
            pass
        
        print("Trading stopped.")
    
    def get_performance(self):
        """Get basic performance metrics"""
        try:
            if hasattr(self.dwx, 'account_info'):
                return {
                    'balance': self.dwx.account_info.get('balance', 0),
                    'equity': self.dwx.account_info.get('equity', 0),
                    'trade_count': self.trade_count
                }
        except:
            return {'error': 'Could not retrieve performance data'}

def main():
    # Configuration
    MT4_FILES_PATH = "C:/Users/YourUsername/AppData/Roaming/MetaQuotes/Terminal/YourBrokerID/MQL4/Files"
    MODEL_PATH = "models/gbpusd_dqn.pth"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run train_dqn.py first to create the model.")
        return
    
    # Create and start trader
    trader = LiveTrader(MT4_FILES_PATH, MODEL_PATH)
    trader.start_trading()

if __name__ == "__main__":
    main()
```

## Component 4: Testing and Validation

### File: `backtest_validator.py`
```python
import pandas as pd
import numpy as np
import torch
from train_dqn import DQNAgent, prepare_data
from gym_trading_env.environments import TradingEnv
import matplotlib.pyplot as plt

def backtest_model(model_path, test_data_start='2018-01-01'):
    """Backtest the trained model on out-of-sample data"""
    
    # Load and prepare data
    df = prepare_data()
    
    # Split data
    train_data = df[df.index < test_data_start]
    test_data = df[df.index >= test_data_start]
    
    print(f"Training data: {len(train_data)} samples")
    print(f"Testing data: {len(test_data)} samples")
    
    if len(test_data) < 100:
        print("Not enough test data")
        return
    
    # Create test environment
    test_env = TradingEnv(
        df=test_data,
        positions=[-1, 0, 1],
        reward_function=lambda hist: np.log(hist["portfolio_valuation", -1] / hist["portfolio_valuation", -2]),
        trading_fees=0.0001,
        portfolio_initial_value=10000
    )
    
    # Load trained model
    state_size = test_env.observation_space.shape[0]
    action_size = test_env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.q_network.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.epsilon = 0  # No exploration during testing
    
    # Run backtest
    state = test_env.reset()[0]
    total_reward = 0
    portfolio_values = []
    actions = []
    
    while True:
        action = agent.act(state)
        next_state, reward, done, truncated, info = test_env.step(action)
        
        portfolio_values.append(info['portfolio_valuation'])
        actions.append(action)
        total_reward += reward
        state = next_state
        
        if done or truncated:
            break
    
    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(np.log(portfolio_values))
    
    metrics = {
        'total_return': (portfolio_values[-1] / portfolio_values[0] - 1) * 100,
        'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
        'max_drawdown': calculate_max_drawdown(portfolio_values),
        'total_trades': len([i for i in range(1, len(actions)) if actions[i] != actions[i-1]])
    }
    
    print("\nBacktest Results:")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.ylabel('Portfolio Value')
    
    plt.subplot(2, 1, 2)
    plt.plot(actions)
    plt.title('Trading Actions Over Time')
    plt.ylabel('Action (0=Short, 1=Hold, 2=Long)')
    plt.xlabel('Time Steps')
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    plt.show()
    
    return metrics

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown percentage"""
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak * 100
    return np.max(drawdown)

if __name__ == "__main__":
    model_path = "models/gbpusd_dqn.pth"
    metrics = backtest_model(model_path)
```

## File Structure

```
FX/
├── new-ideas/
│   └── new-idea-plan.md          # This file
├── models/
│   └── gbpusd_dqn.pth           # Trained model (created by training)
├── data/
│   ├── GBPUSD60.csv             # Your existing data
│   └── GBPUSD240.csv
├── training/
│   ├── train_dqn.py             # Training script
│   └── backtest_validator.py    # Validation script
├── live_trading/
│   ├── trading_bridge.py        # Bridge module
│   ├── live_trader.py           # Live trading interface
│   └── dwxconnect/              # DWXConnect project files
│       ├── python/
│       │   └── api/
│       │       └── dwx_client.py
│       └── mql/
│           └── DWX_Server_MT4.mq4
└── gym_trading_env/             # Gym-Trading-Env project files
    └── src/
        └── gym_trading_env/
```

## Implementation Steps

### Phase 1: Training (Week 1)
1. Set up gym-trading-env environment
2. Run `train_dqn.py` with your GBPUSD data
3. Validate with `backtest_validator.py`
4. Iterate on features/parameters until reasonable performance

### Phase 2: Bridge Development (Week 2)  
1. Implement `trading_bridge.py`
2. Test state calculation with historical data
3. Verify action mapping logic
4. Unit test all components

### Phase 3: Live Integration (Week 3)
1. Set up DWXConnect with demo MT4 account
2. Implement `live_trader.py`
3. Test with paper trading (small positions)
4. Monitor for 1 week before real money

### Phase 4: Optimization (Week 4+)
1. Improve features based on live performance
2. Add risk management (stop-loss, position sizing)
3. Implement performance monitoring
4. Scale up position sizes gradually

## Key Testing Points

1. **State Consistency**: Verify bridge creates same features as training
2. **Action Execution**: Confirm DWX commands execute correctly
3. **Risk Limits**: Test position sizing and stop-loss functionality
4. **Connection Stability**: Monitor MT4 connection reliability
5. **Performance Tracking**: Log all trades and calculate metrics

## Risk Management

- Start with 0.01 lot sizes
- Maximum 1 open position at a time  
- Stop trading if drawdown > 10%
- Manual override to close all positions
- Demo account testing for minimum 2 weeks

## Success Metrics

- **Training**: Positive Sharpe ratio > 0.5 on out-of-sample data
- **Live Trading**: Consistent execution without errors
- **Performance**: Break-even or better in first month
- **Risk**: Maximum drawdown < 5% with small position sizes

This implementation plan provides a complete, testable system that connects the two GitHub projects with minimal complexity while maintaining the ability to train, validate, and execute trades automatically.