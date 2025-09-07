# Snake Trading AI: Forex DQN System

A Snake-inspired Forex trading AI system using Deep Q-Learning (DQN) that applies game mechanics to financial markets: minimal state representation, binary rewards, and simple action spaces for robust trading decisions.

## Philosophy: Simplicity Through Game Theory

This system tests the hypothesis that financial markets behave like sophisticated random walks, where the edge comes from superior **risk management** and **timing** rather than complex prediction models. By borrowing from Snake AI's success with minimal inputs and immediate feedback, we achieve consistent performance through simplicity.

## Core Trading Principles

### The 5-3-1 Rule
- **5 Currency Pairs**: Focus on major liquid pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD)
- **3 Strategies**: Price action, hedging system, and range completion
- **1 Session**: Optimal trading during London/NY overlap (8-12 EST)

### Risk Management Framework
- **2% Rule**: Never risk more than 2% of total capital on a single trade
- **Fixed Position Size**: 0.01 lots to eliminate bet sizing complexity
- **Hedging System**: Automatic 30-pip opposite stops with 40-pip profit targets
- **Binary Rewards**: Clear +10/-10 signals eliminate ambiguous feedback

## System Architecture

### Snake-Inspired Design
```
State Space (4 features):
├── Price Momentum: Normalized price change over 5 periods
├── Position State: Current direction (long/short/flat)
├── Unrealized PnL: Position profit/loss ratio  
└── Time Factor: Session time normalization (0-1)

Action Space (3 actions):
├── 0: Close/Hold — Close position or hold if flat
├── 1: Long — Enter long position (if currently flat)
└── 2: Short — Enter short position (if currently flat)

Reward System:
├── +10: Profitable trade closure
├── -10: Losing trade closure
└── 0: While position is open
```

### Model Architecture
```python
# Linear DQN: 4 inputs → 256 hidden → 3 outputs
Linear_QNet(4, 256, 3)

# Training Parameters
MAX_MEMORY = 100_000    # Experience replay buffer
BATCH_SIZE = 1000       # Training batch size
LR = 0.001             # Learning rate
gamma = 0.9            # Discount factor
```

## Current System Components

### Core Modules
- **Agent** (`core/agent.py`): DQN agent with epsilon-greedy exploration, experience replay
- **Environment** (`game_env/forex_env.py`): Trading environment with Snake-like state representation
- **Model** (`core/model.py`): Linear DQN with PyTorch backend
- **Data Feed** (`integration/data_feed.py`): CSV processing with temporal constraints

### Trading Integration
- **MT5 Connector** (`integration/dwx_connector.py`): MetaTrader 5 bridge for live trading
- **Signal Translator** (`integration/signal_translator.py`): DQN actions → MT5 orders
- **Risk Manager** (`utils/risk_management.py`): Position sizing and stop loss logic

### Training System
- **Offline Training** (`training/trainer.py`): Batch learning from historical data
- **Threading Model**: F1 (sync) → F2 (logic) → F3 (execution) prevents look-ahead bias
- **Evaluation** (`training/evaluator.py`): Model validation and performance metrics

## Installation & Setup

### Dependencies
```bash
pip install torch pandas numpy matplotlib pathlib
```

### Data Preparation
Prepare CSV files with columns: `timestamp, open, high, low, close, volume`

### Basic Training
```bash
# Train with historical data
python run.py --csv path/to/gbpusd_h1.csv --mode sequential

# Experimental threaded mode
python run.py --csv path/to/gbpusd_h1.csv --mode threaded
```

### Testing
```bash
# Run all tests
python -m unittest discover tests/

# Specific test modules  
python -m unittest tests.test_agent
python -m unittest tests.test_env
python -m unittest tests.test_trainer
```

## Advanced Hedging Strategy

### Automatic Hedge System
The system implements a sophisticated hedging mechanism that replaces traditional stop losses:

```python
# Core Hedging Parameters
HEDGE_DISTANCE = 30     # pips - opposite stop placement
INNER_TP_THRESHOLD = 40 # pips - profit target for hedge side
POCKET_AMOUNT = 100     # fixed profit amount to retain
```

### Hedge Execution Flow
1. **Entry + Auto-Hedge**: Place main trade → set opposite stop 30 pips away
2. **Hedge Activation**: If price reverses, automatically enter opposite position
3. **Profit Trimming**: When hedge side reaches +40 pips, close it and pocket fixed amount
4. **Position Reduction**: Apply remaining profit to reduce losing outer position size
5. **Re-hedge**: Set new 30-pip opposite stop for remaining position size
6. **Squeeze/Trail**: Move pending stops closer to maintain 30-pip gap

### Benefits Over Traditional Stops
- **Smoother Equity**: Reduces large drawdowns via hedged positions
- **Time Recovery**: Buys time for mean reversion without realizing losses
- **Systematic Rules**: Eliminates emotional decision making
- **Capital Recycling**: Converts hedge profits into position size reduction

## Project Structure

```
snake-trading/
├── README.md
├── CLAUDE.md                    # Development guidelines
├── run.py                       # Main entry point
├── requirements.txt
├── core/
│   ├── agent.py                # DQN agent implementation
│   ├── model.py                # Neural network architecture
│   ├── config.py               # Configuration constants
│   └── replay_buffer.py        # Experience replay memory
├── game_env/
│   ├── forex_env.py           # Main trading environment
│   ├── base_env.py            # Base environment class
│   └── hierarchical_env.py    # Multi-timeframe environment
├── integration/
│   ├── data_feed.py           # CSV data processing
│   ├── dwx_connector.py       # MetaTrader 5 integration
│   └── signal_translator.py   # Action → order conversion
├── training/
│   ├── trainer.py             # Training loops
│   └── evaluator.py           # Performance evaluation
├── utils/
│   ├── logging.py             # Training logs
│   ├── metrics.py             # Performance metrics
│   ├── plots.py               # Visualization tools
│   └── risk_management.py     # Risk controls
└── tests/
    ├── test_agent.py
    ├── test_env.py
    └── test_trainer.py
```

## Performance Considerations

### Current System
- **GPU Support**: Automatic CUDA usage when available
- **Memory Efficiency**: Bounded replay buffer prevents memory leaks
- **Threading**: Sequential mode recommended for stability
- **Temporal Constraints**: Prevents look-ahead bias in training

### Training Timeline
- **Convergence**: Typically 50,000-100,000 episodes for stable performance
- **Epsilon Decay**: Starts at 80, decreases with experience
- **Model Persistence**: Auto-saves when record performance achieved

## Future Roadmap: Multi-Pair Portfolio System

### Vision: Correlation-Aware Trading
Plans are in development for a sophisticated multi-currency system featuring:

- **8 Major Pairs**: Simultaneous trading across EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, EURGBP, EURJPY
- **56D State Space**: 48 pair-specific features + 8 portfolio-level features
- **Correlation Engine**: Real-time correlation matrix with conflict detection
- **Portfolio Risk**: 3% maximum total exposure across all positions
- **Currency Balance**: <40% exposure to any single base currency

### Advanced Risk Systems
- **Correlation Circuit Breaker**: Auto-close all positions during correlation spikes
- **Session Coordination**: Optimal timing across multiple currency pairs
- **Emergency Systems**: Comprehensive connection loss and crisis management

*Note: Multi-pair system is in research phase. Current single-pair system is production-ready.*

## Critical Warnings

### Risk Factors
- **Capital Requirements**: Minimum $10,000 for meaningful testing
- **Market Conditions**: Performance varies across different volatility regimes
- **Technology Dependencies**: Requires stable data feed and execution platform
- **Correlation Risk**: Hedging effectiveness can break down during market crises

### Development Notes
- **Paper Trading**: Extensive backtesting recommended before live deployment
- **Position Sizing**: Start with minimum lots until system validation
- **Market Hours**: System optimized for major session overlaps
- **Data Quality**: Requires clean, gap-free price data for training

## Contributing

1. Follow existing code patterns and architectural decisions
2. Maintain Snake-inspired simplicity principles
3. Add comprehensive tests for new components
4. Update documentation for significant changes

## License

MIT License - see LICENSE file for details.

---

*"The snake grows through patience and precision, not speed and complexity."*