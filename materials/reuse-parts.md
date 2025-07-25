## **Reusable Components from original_src.txt**

### **agent.py**
```pseudocode
Class Agent:
    __init__(config_name) // Keep basic structure
    remember(state, action, reward, next_state, done) // Keep as-is
    train_long_memory() // Keep replay buffer logic
    train_short_memory() // Keep immediate training
    get_action(state) // Keep epsilon-greedy, remove adaptive logic
    save_model(filename) // Keep checkpointing
    load_model(filename) // Keep model loading
```

### **model.py**
```pseudocode
Class Linear_QNet:
    __init__(input_size=4, output_size=2) // Modify dimensions
    forward(x) // Keep basic forward pass
    save(file_name) // Keep model persistence
    load(file_name) // Keep model loading

Class QTrainer:
    train_step(state, action, reward, next_state, done) // Keep core training
    // Remove: double DQN, dueling DQN, target networks
```

### **forex_game.py**
```pseudocode
Class ForexGameAI:
    __init__(data, initial_balance) // Keep basic structure
    reset() // Keep episode reset
    get_portfolio_value() // Keep balance tracking
    get_current_price() // Keep price access
    current_step // Keep step counter
    position tracking // Keep basic position state
    trades_history // Keep trade logging
    
    // Remove: all _calculate_* methods, hierarchical logic, complex features
```

### **real_data_loader.py**
```pseudocode
load_real_gbpusd_data(h1_path, start_samples, total_samples) // Keep as-is
get_data_statistics(data) // Keep data validation
// Keep: data filtering, time parsing, boundary handling
```

### **helper.py**
```pseudocode
plot(scores, mean_scores, portfolio_values) // Keep basic plotting
plot_trade_analysis(trades_history) // Keep trade visualization
// Remove: complex matplotlib configurations
```

### **trainer.py**
```pseudocode
Class V4Trainer:
    load_data() // Keep data loading logic
    train(max_episodes, save_interval) // Keep training loop structure
    save_checkpoint() // Keep checkpoint system
    load_checkpoint(checkpoint_path) // Keep resume functionality
    
    // Remove: parallel training, grid search integration
```

## **Components to Discard**

```pseudocode
// config.py - entire file (too complex)
// grid_search_*.py - entire files
// parallel_*.py - entire files  
// dwx_integration.py - entire file (premature)
// debug_hierarchical.py - entire file
// gym_env_adapter.py - entire file
// simple_risk_manager.py - too complex, rebuild simple

// From forex_game.py:
_calculate_volume_ratio()
_calculate_volume_price_confirmation()
_get_session_indicator()
_calculate_volatility_clustering()
_calculate_intrabar_momentum()
_calculate_sr_distance()
_calculate_atr()
_detect_market_regime()
set_sl_tp_levels()
check_sl_tp_hit()
calculate_position_size()
```

## **Modification Requirements**

```pseudocode
// Change input dimensions: 13 → 4
Linear_QNet(input_size=4, hidden_size=256, output_size=2)

// Simplify get_state() to return only:
[price_momentum, position_state, unrealized_pnl, time_factor]

// Simplify play_step() to:
Binary actions: 0=Short, 1=Long
Binary rewards: +10/-10
Single termination: end_of_data

// Remove from agent.py:
adaptive epsilon logic
performance tracking
config management
```