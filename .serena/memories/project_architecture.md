# Snake Trading AI - Project Architecture

## Project Purpose
Snake-inspired Forex trading AI system using Deep Q-Learning (DQN) for trading decisions. Applies Snake game principles to financial markets: minimal state representation (4 features), binary rewards (+10/-10), simple action space (Close/Long/Short).

## Tech Stack
- **Deep Learning**: PyTorch for DQN model
- **Data Processing**: Pandas, NumPy
- **Technical Analysis**: TA-Lib for financial indicators
- **Environment**: Custom Gym-like trading environment
- **Visualization**: Matplotlib for plotting
- **Testing**: Python unittest framework
- **Python Version**: Python 3 with virtual environment

## Code Structure

### Core Modules
- `core/agent.py` - DQN agent with epsilon-greedy exploration, experience replay
- `core/model.py` - Linear DQN architecture (4→256→3), QTrainer for backprop  
- `core/config.py` - Central configuration constants
- `core/replay_buffer.py` - Experience replay memory management

### Environment
- `game_env/forex_env.py` - Main trading environment with Snake-like state representation
- `game_env/base_env.py` - Base environment class
- `game_env/hierarchical_env.py` - Multi-timeframe environment
- `game_env/gym_adapter.py` - Gym wrapper compatibility

### Integration
- `integration/data_feed.py` - CSV processing with threading for temporal constraints
- `integration/dwx_connector.py` - MetaTrader connection for live trading
- `integration/signal_translator.py` - Action to MT4 order conversion

### Training & Utils
- `training/trainer.py` - Offline training loops
- `training/evaluator.py` - Model evaluation and metrics
- `utils/` - Logging, metrics, plotting, risk management utilities

### Entry Point
- `run.py` - Main CLI interface with --csv and --mode arguments

## Key Architecture Decisions

### Snake-Inspired Design
- **4-feature state**: Price momentum, position state, unrealized PnL, time factor
- **3 actions**: 0=Close/Hold, 1=Long, 2=Short  
- **Binary rewards**: +10 profit, -10 loss, 0 while open
- **Fixed lot size**: 0.01 to remove bet sizing complexity

### Threading Model
- Sequential vs threaded processing modes
- Temporal constraints to prevent look-ahead bias
- Thread synchronization for realistic trading simulation

### Modular Design
- Clean separation between training and live trading
- Reusable components across different environments
- Bridge modules for live trading integration without code duplication

## Current Implementation Status

### ✅ Implemented Components
- **helpers.py**: Comprehensive data processing library
  - TechnicalIndicators: RSI, MACD, Bollinger Bands via TA-Lib
  - RollingFeatures: Moving averages, momentum, statistical features
  - AdvancedFeatures: Multiple volatility models (Parkinson, Garman-Klass, Yang-Zhang)
  - PivotPoints: Standard, Woodie, Camarilla calculations
  - FibonacciLevels: Standard and extended retracement levels
  - PriceTransformations: OHLC patterns, candlestick recognition
  - DataProcessor: Main orchestrator class with JSON configuration support

- **ohlcv_feeder.py**: Real-time data streaming simulation
  - CSV parsing with configurable formats and delimiters
  - Speed multiplier for accelerated backtesting
  - Lookback windows for historical context
  - Threading support for background data feeding
  - State management and data queuing

- **original-src/**: Reference Snake DQN implementation
  - Complete Snake game with pygame
  - DQN agent with experience replay
  - Linear neural network (11→256→3)
  - Training loop with epsilon decay

### 🚧 Not Yet Implemented
- Core DQN trading system
- Forex trading environment
- Training infrastructure
- Live trading integration
- Test suites
- Formal dependency management

## Design Patterns Used

### Factory Pattern
- DataProcessor class instantiates various feature generators
- Configuration-driven feature selection

### Template Method
- Base processing pattern with customizable feature addition
- Standardized DataFrame → processed DataFrame pipeline

### Observer Pattern
- Threading in ohlcv_feeder for data events
- Callback functions for data streaming events

### Strategy Pattern
- Multiple volatility calculation methods
- Configurable pivot point calculation types

### Builder Pattern
- Incremental DataFrame feature building
- Method chaining for data processing pipeline

## Data Flow Architecture

### Input Pipeline
```
CSV Files → ohlcv_feeder.py → Streaming Simulation → DataProcessor → Feature Engineering → Trading Environment (future)
```

### Feature Engineering Pipeline
```
Raw OHLCV → Technical Indicators → Rolling Features → Advanced Features → Pattern Recognition → Categorical Features → ML-Ready DataFrame
```

### Future Trading Pipeline (Planned)
```
Features → DQN Agent → Action Decision → Environment Execution → Reward Calculation → Experience Replay → Model Training
```

## Configuration Management

### Current Approach
- JSON-based configuration files
- Dictionary-driven feature selection
- Default parameter fallbacks
- Flexible input handling (file paths or direct dictionaries)

### Configuration Structure
```python
{
    "technical_indicators": {
        "RSI": {"time_periods": [14, 21], "input_columns": ["close"]},
        "MACD": {"input_columns": ["close"], "output_columns": ["_macd", "_signal", "_hist"]}
    },
    "rolling_features": {
        "columns": ["close", "volume"],
        "windows": [5, 10, 20, 50],
        "functions": ["mean", "std", "max", "min"]
    }
}
```

## Future Architecture Evolution

### Multi-Pair System (Planned)
According to materials/new_plan.md:
- **8 major currency pairs**: EURUSD, GBPUSD, USDJPY, etc.
- **56-dimensional state space**: 48 pair-specific + 8 portfolio features
- **28 actions**: 24 pair-specific + 4 portfolio-wide
- **Correlation awareness**: Real-time correlation matrix
- **Portfolio risk management**: 3% maximum exposure

### Scalability Considerations
- Modular component architecture supports future expansion
- Configuration-driven approach enables easy feature modification
- Threading model prepares for multi-pair simultaneous processing
- Clean separation between data processing and trading logic

## Development Phases

### Phase 1: Core DQN Infrastructure (Next)
1. Implement basic forex trading environment
2. Create DQN agent and neural network models  
3. Build training loop infrastructure

### Phase 2: Single-Pair Trading
1. Implement 4D state space (price momentum, position, PnL, time)
2. 3-action space (Close/Long/Short)
3. Binary reward system (+10/-10)

### Phase 3: Multi-Pair System
1. Expand to 56D state space across 8 pairs
2. Implement correlation detection  
3. Add portfolio-level risk management

## Technical Debt and Improvements

### Immediate Needs
- Add comprehensive unit tests
- Implement formal dependency management
- Add docstrings to all public methods
- Set up code formatting and linting

### Long-term Architecture Goals
- Microservices architecture for production deployment
- Event-driven architecture for real-time trading
- Database integration for historical data storage
- REST API for external system integration