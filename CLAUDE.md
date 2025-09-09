# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Snake-inspired Forex trading AI project that is currently in the foundation/planning stage. The goal is to develop a multi-pair forex trading system using Deep Q-Learning (DQN) with Snake game principles: minimal state representation, binary rewards, and simple action spaces.

**Current Status**: Infrastructure and planning phase - core trading system components are not yet implemented.

## Current Architecture

The project currently consists of:

### Implemented Components
- **Data Processing** (`helpers.py`): Comprehensive OHLCV data processing library with technical indicators, rolling features, volatility calculations, pivot points, Fibonacci levels, and advanced features
- **Data Streaming** (`ohlcv_feeder.py`): Real-time OHLCV feeder that simulates live streaming from CSV files with configurable speed multipliers
- **Original Snake Reference** (`original-src/`): Original Snake game DQN implementation from tutorial series for reference
- **Project Planning** (`materials/`): Detailed planning documents for the future multi-pair trading system

### Not Yet Implemented
- DQN agent and neural network models
- Trading environment
- Training infrastructure
- Live trading integration
- Test suites

## Development Commands

### Basic Data Processing Test
```bash
# Test the OHLCV feeder with sample data
python ohlcv_feeder.py
```

### Dependencies
Current dependencies based on existing code:
```bash
pip install pandas numpy talib
```

Additional dependencies will be needed when implementing the DQN system:
```bash
pip install torch matplotlib pathlib
```

## Key Design Principles

### Snake-Inspired Philosophy (Planned)
- **Minimal State**: Limited feature set vs complex technical indicators
- **Binary Rewards**: Clear +10/-10 signals vs continuous reward functions
- **Fixed Position Size**: Remove bet sizing complexity
- **Simple Actions**: Close/Long/Short only

### Multi-Pair Vision (Future)
According to `materials/new_plan.md`, the system will evolve to:
- **8 Major Pairs**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, EURGBP, EURJPY
- **56D State Space**: 48 pair-specific features + 8 portfolio-level features
- **28 Actions**: 24 pair-specific + 4 portfolio-wide actions
- **Correlation Awareness**: Real-time correlation matrix and conflict detection

## Current File Structure

```
snake-trading/
├── helpers.py                   # Data processing utilities (IMPLEMENTED)
├── ohlcv_feeder.py             # CSV streaming simulator (IMPLEMENTED)
├── original-src/               # Snake tutorial reference code
│   ├── README_snake.md
│   ├── game.py
│   ├── model.py
│   ├── agent.py
│   └── ...
├── materials/                  # Planning documents
│   ├── new_plan.md            # Multi-pair system design
│   └── content.md
├── README.md                   # Project documentation
└── Old_CLAUDE.md              # Previous documentation (outdated)
```

## Data Processing Capabilities

### Technical Indicators
The `helpers.py` module provides extensive technical analysis capabilities:
- **Technical Indicators**: RSI, MACD, Bollinger Bands, etc. (via TA-Lib)
- **Rolling Features**: Moving averages, standard deviations, momentum
- **Price Patterns**: Doji, hammer, shooting star detection
- **Volatility Models**: Parkinson, Garman-Klass, Yang-Zhang volatility
- **Pivot Points**: Standard, Woodie, Camarilla calculations
- **Fibonacci Levels**: Standard and extended retracements

### OHLCV Data Streaming
The `ohlcv_feeder.py` provides:
- CSV file parsing with configurable formats
- Real-time simulation with speed multipliers
- Lookback windows for historical context
- State management for current market position

## Development Roadmap

Based on planning documents, the next development phases are:

### Phase 1: Core DQN Infrastructure
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

## Testing Strategy

Currently no tests exist. When implementing:
- Unit tests for data processing components
- Environment state transition tests
- Model training validation
- Backtesting on historical CSV data

## Important Notes

- **Data Requirements**: OHLCV CSV files with columns: timestamp, open, high, low, close, volume
- **Development Stage**: Foundation components only - main trading system not yet built
- **Reference Code**: Use `original-src/` for Snake DQN implementation patterns
- **Planning**: Refer to `materials/new_plan.md` for detailed future architecture
- **Data Processing**: `helpers.py` provides production-ready feature engineering capabilities