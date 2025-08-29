# Suggested Commands for Snake Trading AI

## Core Development Commands

### Training
```bash
# Basic training with CSV data (recommended)
python run.py --csv path/to/gbpusd_h1.csv --mode sequential

# Experimental threaded mode
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

# Using pytest (if available)
python -m pytest tests/
python -m pytest tests/test_agent.py -v
```

### Dependencies Installation
```bash
# Install required packages
pip install torch pandas numpy matplotlib pathlib

# For development, also need
pip install pytest  # for testing
```

### Model Management
```bash
# Models are automatically saved to ./model/model.pth when record is beaten
# No manual model management commands - handled by training loop
```

## System Utilities (Linux)
```bash
# Basic file operations
ls          # list files
cd          # change directory
grep        # search text
find        # find files
git         # version control
```

## Notes
- No linting/formatting tools configured (flake8, black not available)
- No build commands needed (pure Python project)  
- No config files for development tools
- Testing uses standard unittest framework
- Project uses standard Python 3 with virtual environment at /home/hung/env/.venv/