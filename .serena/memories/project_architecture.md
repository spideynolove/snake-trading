# Snake Trading AI - Project Architecture

## Project Purpose
Snake-inspired Forex trading AI system using Deep Q-Learning (DQN) for trading decisions. Applies Snake game principles to financial markets: minimal state representation (4 features), binary rewards (+10/-10), simple action space (Close/Long/Short).

## Tech Stack
- **Deep Learning**: PyTorch for DQN model
- **Data Processing**: Pandas, NumPy
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