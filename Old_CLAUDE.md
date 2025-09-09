# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Snake-inspired Forex trading AI system using Deep Q-Learning (DQN) for trading decisions. The system applies Snake game principles to financial markets: minimal state representation (4 features), binary rewards (+10/-10), and simple action space (Close/Long/Short).

## Core Architecture

### Key Components
- **Agent** (`core/agent.py`): DQN agent with epsilon-greedy exploration, experience replay
- **Environment** (`game_env/forex_env.py`): Trading environment with Snake-like state representation
- **Model** (`core/model.py`): Linear DQN (4 inputs → 256 hidden → 3 outputs)
- **Data Feed** (`integration/data_feed.py`): CSV processing with threading for temporal constraints
- **Training** (`training/`): Offline training loops and evaluation
- **Utils** (`utils/`): Logging, metrics, plotting, risk management

### State Representation (4 features like Snake)
1. **Price Momentum**: Normalized price change over last 5 periods
2. **Position State**: Current position direction (long/short/flat)  
3. **Unrealized PnL**: Current position profit/loss ratio
4. **Time Factor**: Session time normalization (0-1)

### Action Space
- **0: Close/Hold** — Close position or hold if flat
- **1: Long** — Enter long position (if currently flat)
- **2: Short** — Enter short position (if currently flat)

### Reward System
- **+10**: Profitable trade closure
- **-10**: Losing trade closure  
- **0**: While position is open

## Development Commands

### Training
```bash
# Basic training with CSV data
python run.py --csv path/to/gbpusd_h1.csv --mode sequential

# Threaded mode (experimental)  
python run.py --csv path/to/gbpusd_h1.csv --mode threaded
```

### Testing
```bash
# Run all tests
python -m unittest discover tests/

# Run specific test modules
python -m unittest tests.test_agent
python -m unittest tests.test_env
python -m unittest tests.test_trainer
```

### Dependencies
Install required packages:
```bash
pip install torch pandas numpy matplotlib pathlib
```

## Key Design Principles

### Snake-Inspired Simplicity
- **Minimal State**: Only 4 features vs complex technical indicators
- **Binary Rewards**: Clear +10/-10 signals vs continuous reward functions
- **Fixed Position Size**: 0.01 lots to remove bet sizing complexity
- **Simple Termination**: End of data or max steps

### Threading Model
The system uses a sophisticated threading model to prevent look-ahead bias:
- **F1 Thread**: Data synchronization event
- **F2 Thread**: Trading logic processing  
- **F3 Thread**: Order execution
- **Temporal Constraints**: Ensures agents only see current data point

### Modular Architecture
```
core/           # Agent, model, replay buffer, config
game_env/       # Trading environments (base, forex, hierarchical)
integration/    # Live trading integration (DWX, data feeds)
training/       # Training loops, evaluation, grid search
utils/          # Logging, metrics, plotting, risk management
tests/          # Unit tests for all components
```

## Configuration

### Model Parameters (in core/agent.py)
- `MAX_MEMORY = 100_000` - Experience replay buffer size
- `BATCH_SIZE = 1000` - Training batch size  
- `LR = 0.001` - Learning rate
- `gamma = 0.9` - Discount factor
- Input size: 4, Hidden: 256, Output: 3

### Environment Parameters
- `initial_balance = 10000.0` - Starting capital
- Fixed lot size: 0.01 (implicit in reward calculation)
- Session time normalization: 24-hour cycles

## Live Trading Integration

The system includes bridge modules for live trading via DWX (MetaTrader integration):
- `integration/dwx_connector.py` - MetaTrader connection
- `integration/signal_translator.py` - Action to MT4 order conversion
- `integration/data_feed.py` - Real-time data processing

## Common Development Tasks

### Adding New Features
1. Modify state representation in `game_env/forex_env.py:get_state()`
2. Update model input size in `core/model.py` and `core/agent.py`
3. Add corresponding tests in `tests/`

### Training New Models
1. Prepare CSV data with columns: timestamp, open, high, low, close, volume
2. Run training with appropriate CSV path
3. Models auto-save to `./model/model.pth` when record is beaten

### Debugging Training Issues
- Check data validation in `integration/data_feed.py`
- Monitor epsilon decay in agent (starts at 80, decreases with games)
- Use `utils/plots.py` for training visualizations
- Check threading synchronization if using threaded mode

## Performance Considerations

- **GPU Support**: PyTorch models automatically use CUDA if available
- **Memory Management**: Experience replay buffer has max size limit
- **Threading**: Sequential mode is more stable, threaded mode is experimental
- **Data Processing**: CSV files are validated and normalized on load

## Testing Strategy

Tests cover:
- Agent initialization and decision making
- Environment state transitions and rewards
- Model forward/backward passes
- Data feed validation and processing
- Memory replay functionality

## Important Notes

- The system prioritizes simplicity over complexity (Snake philosophy)
- Temporal constraints prevent look-ahead bias in training
- Binary rewards create clear learning signals
- Fixed position sizing reduces parameter space
- Modular design allows easy experimentation and extension