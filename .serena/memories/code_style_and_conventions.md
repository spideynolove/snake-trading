# Code Style and Conventions

## Naming Conventions
- **Classes**: PascalCase (e.g., `ForexEnv`, `Linear_QNet`, `QTrainer`)
- **Methods/Functions**: snake_case (e.g., `get_state`, `train_long_memory`, `step`)
- **Variables**: snake_case (e.g., `current_step`, `initial_balance`, `n_games`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_MEMORY`, `BATCH_SIZE`, `ACTION_CLOSE`)

## Import Organization
- Standard library imports first (e.g., `import os`, `import random`)
- Third-party imports second (e.g., `import torch`, `import pandas`, `import numpy`)
- Local imports last (e.g., `from core.agent import Agent`)

## Configuration Management
- All configuration constants centralized in `core/config.py`
- No hardcoded magic numbers in implementation files
- Clear constant names with descriptive prefixes (e.g., `REWARD_PROFIT`, `MODEL_INPUT_SIZE`)

## Code Patterns

### Error Handling
- Try-catch blocks for data loading and episode processing
- Graceful failure with informative error messages
- Example: `run.py` handles CSV loading errors and training failures

### Threading Safety
- Decision locks for thread-safe agent actions (`self.decision_lock`)
- Temporal constraints class for preventing look-ahead bias

### Memory Management
- Deque with maxlen for experience replay buffer
- Automatic memory cleanup when exceeding `MAX_MEMORY` limit

### Model Persistence  
- Automatic model saving when records are beaten
- Standard PyTorch state_dict saving/loading pattern
- Models saved to `./model/model.pth`

## Testing Conventions
- Standard unittest framework
- Test classes inherit from `unittest.TestCase`
- Test methods prefixed with `test_`
- Setup method `setUp()` for common test data
- Mock data using random generation for reproducible tests

## Documentation Style
- Class and method docstrings not consistently used
- Inline comments for complex logic
- Configuration constants are self-documenting

## Type Hints
- Not consistently used throughout the codebase
- NumPy arrays and PyTorch tensors used without explicit typing

## File Organization
- Modular structure with clear separation of concerns
- Single responsibility principle for classes and modules
- Utility functions grouped by functionality in `utils/`