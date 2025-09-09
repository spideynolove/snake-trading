# Snake Trading AI - Code Style and Conventions

## Naming Conventions

### Classes
- **PascalCase**: `DataProcessor`, `RealTimeOHLCVFeeder`, `TechnicalIndicators`
- **Descriptive names**: Classes clearly indicate their purpose
- **No abbreviations**: Full words preferred over abbreviations

### Methods and Functions
- **snake_case**: `process_dataframe()`, `get_current_state()`, `add_technical_indicators()`
- **Descriptive verbs**: Methods start with action verbs (get, add, calculate, process)
- **Private methods**: Prefix with underscore `_load_all_data()`, `_update_processed_data()`

### Variables
- **snake_case**: `current_index`, `speed_multiplier`, `data_file`
- **Descriptive**: Variable names clearly indicate content/purpose
- **No single letters**: Except for loop counters (`i`, `j`) and mathematical formulas

### Constants
- **UPPER_SNAKE_CASE**: `MAX_MEMORY`, `BATCH_SIZE`, `LR` (from original-src)
- **Module level**: Defined at top of files

## Type Hints

### Current Usage
- **Partial adoption**: Type hints used in key methods like `DataProcessor.process_dataframe()`
- **Standard types**: `List[str]`, `Dict[str, Dict]`, `Union[int, str]`
- **pandas types**: `pd.DataFrame` for data processing functions
- **Import pattern**: `from typing import List, Dict, Union, Any`

### Method Signatures
```python
def add_technical_indicators(df: pd.DataFrame, indicators: Dict[str, Dict]) -> pd.DataFrame:
def add_rolling_functions(df: pd.DataFrame, column_names: List[str], 
                        window_sizes: List[Union[int, str]], 
                        functions: List[str]) -> pd.DataFrame:
```

## Docstrings

### Current State
- **Minimal**: Most methods lack docstrings
- **Missing**: No class-level docstrings
- **Inconsistent**: Some methods have brief inline comments instead

### Recommended Style
- Use triple quotes for multi-line docstrings
- Follow Google or NumPy docstring style
- Include parameters, return types, and examples for complex methods

## Import Organization

### Pattern Used
```python
# Standard library imports
import json
import sys
from datetime import datetime
from pathlib import Path

# Third-party imports  
import pandas as pd
import numpy as np

# Local imports
from helpers import DataProcessor
```

### Conventions
- **Standard library first**: Built-in modules
- **Third-party second**: External packages (pandas, numpy, torch)
- **Local imports last**: Project modules
- **Specific imports**: Prefer `from module import Class` for commonly used items

## Class Structure

### Initialization Pattern
```python
class DataProcessor:
    def __init__(self, config_path=None):
        # Initialize sub-components
        self.tech_indicators = TechnicalIndicators()
        self.rolling_features = RollingFeatures()
        # ... other components
        
        # Load configuration
        self.config = self.load_config(config_path) if config_path else {}
```

### Method Organization
1. **`__init__()`** - Constructor first
2. **Public methods** - Main functionality
3. **Private methods** - Helper functions prefixed with underscore
4. **Static methods** - Utility functions using `@staticmethod`

## Error Handling

### Patterns Observed
- **Try-except blocks**: Used for file operations and data parsing
- **None returns**: Return None for failed operations rather than raising exceptions
- **Validation**: Input validation in parsing methods
- **Graceful degradation**: Continue processing when individual items fail

```python
try:
    # Parse data
    record = self.parse_data_line(line)
except:
    return None  # Graceful failure
```

## Data Processing Conventions

### DataFrame Operations
- **Copy pattern**: Always copy input DataFrame: `df_result = df.copy()`
- **Method chaining**: Avoid excessive chaining, prefer intermediate variables
- **Column naming**: Use descriptive suffixes for calculated columns
- **Null handling**: Explicit `.fillna()` at end of processing pipeline

### Configuration
- **Dictionary-based**: Use dictionaries for configuration
- **Default values**: Provide sensible defaults with `.get()` method
- **Flexible inputs**: Accept both file paths and dictionaries for config

## Threading and Concurrency

### Patterns in ohlcv_feeder.py
- **daemon threads**: `daemon=True` for background operations
- **Queue usage**: `queue.Queue()` for thread-safe data passing
- **State management**: Clear running flags and state variables

## Constants and Magic Numbers

### Current Usage
- **Named constants**: Some constants like `MAX_MEMORY = 100_000`
- **Magic numbers**: Some hardcoded values exist (could be improved)
- **Default parameters**: Extensive use of default parameter values

## Future Recommendations

1. **Add comprehensive docstrings** to all classes and public methods
2. **Implement consistent type hints** across all modules
3. **Set up code formatting** with black or similar tool
4. **Add linting** with flake8 or pylint
5. **Create unit tests** with pytest framework
6. **Add logging** instead of print statements for production code