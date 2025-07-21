# 📦 Refactored Project Structure

```

v4/
├── core/
│   ├── agent.py               # Base RL agent logic (epsilon, memory, training)
│   ├── model.py               # DQN architecture: Linear\_QNet, Dueling, etc.
│   ├── config.py              # Central config registry
│   └── replay\_buffer.py       # Separate prioritized/standard buffer
│
├── env/
│   ├── base\_env.py            # Base class for all ForexEnv variants
│   ├── forex\_env.py           # Basic H1 environment
│   ├── hierarchical\_env.py    # Multi-timeframe environment (D1/H1 logic)
│   └── gym\_adapter.py         # Gym wrapper
│
├── integration/
│   ├── dwx\_connector.py       # Handles live trading via DWX
│   ├── data\_feed.py           # Abstracts loading/syncing price data
│   └── signal\_translator.py   # Converts RL action to MT4-compatible signal
│
├── training/
│   ├── trainer.py             # Offline training loop
│   ├── live\_trainer.py        # Live trading loop
│   ├── grid\_search.py         # Grid search logic
│   ├── config\_generator.py    # Generates training configs
│   └── evaluator.py           # Evaluation and stats logging
│
├── utils/
│   ├── logger.py              # Logging, checkpoints
│   ├── metrics.py             # Reward tracking, win-rate, drawdown, etc.
│   ├── risk.py                # Risk rules and position sizing logic
│   └── plots.py               # Visualization
│
├── tests/
│   ├── test\_env.py
│   ├── test\_agent.py
│   └── test\_trainer.py
│
├── notebooks/                 # For experiments, not core code
│   └── exploration.ipynb
│
├── run.py                     # CLI to launch training, evaluation, or live
└── README.md

````

---

# 📈 Program Flow

```mermaid
flowchart TD
    Start([Start]) --> CLI{run.py CLI?}
    CLI -->|train| Train[Trainer]
    CLI -->|live| Live[LiveTrader]
    CLI -->|grid| Grid[GridSearch]

    Train --> LoadData --> InitEnv
    InitEnv --> InitAgent --> TrainLoop
    TrainLoop --> SaveModel --> End([Done])

    Live --> LoadModel --> LiveEnv --> TickLoop
    TickLoop --> Action --> Signal --> DWX --> TickLoop

    Grid --> GenConfigs --> ParallelTrain --> AnalyzeResults --> End
````

---

# 🧠 Pseudocode Overview

---

### `core/agent.py`

```python
class DQNAgent:
    def __init__(self, config): ...
    def act(state): ...
    def train_step(batch): ...
    def remember(...): ...
    def update_epsilon(): ...
```

---

### `core/model.py`

```python
class DQN(nn.Module):
    def __init__(input_size, output_size, config): ...
    def forward(x): ...
def create_model(config): ...
```

---

### `env/base_env.py`

```python
class BaseForexEnv:
    def reset(): ...
    def step(action): ...
    def get_state(): ...
```

---

### `env/hierarchical_env.py`

```python
class HierarchicalForexEnv(BaseForexEnv):
    def get_state(): ...
    def get_bias_from_d1(): ...
```

---

### `integration/dwx_connector.py`

```python
class DWXConnector:
    def send_signal(action): ...
    def update_risk(): ...
```

---

### `training/trainer.py`

```python
def train(agent, env, config):
    for episode in range(config.episodes):
        state = env.reset()
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(...)
        agent.replay()
```

---

### `training/grid_search.py`

```python
def run_search(configs):
    for config in configs:
        agent = DQNAgent(config)
        result = train(agent, env, config)
    analyze_results()
```

---

### `run.py`

```python
if args.mode == 'train':
    config = load_config(args.config)
    train(agent, env, config)
elif args.mode == 'live':
    run_live_trading(agent, env)
elif args.mode == 'grid':
    run_search(configs)
```