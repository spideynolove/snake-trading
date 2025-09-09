# Snake Trading AI - Suggested Commands

## Basic Development Commands

### Data Processing and Testing
```bash
# Test the OHLCV data feeder with sample data
python ohlcv_feeder.py

# Test the data processing utilities (requires sample CSV data)
python -c "from helpers import DataProcessor; dp = DataProcessor(); print('DataProcessor loaded successfully')"
```

### Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Linux)
source .venv/bin/activate

# Install current dependencies
pip install pandas numpy talib

# Install future dependencies (when implementing DQN)
pip install torch matplotlib pathlib
```

### File Operations
```bash
# List project files
ls -la

# Find Python files
find . -name "*.py" -type f

# Search in Python files
grep -r "class " *.py

# View project structure
tree . -I '__pycache__|.venv|*.pyc'
```

### Git Operations
```bash
# View current status
git status

# View recent commits
git log --oneline -10

# View current branch
git branch

# View changes
git diff

# Stage changes
git add .

# Commit changes
git commit -m "commit message"
```

## Project-Specific Commands

### Exploring Reference Code
```bash
# View the original Snake DQN implementation
ls original-src/

# Run the Snake game test (requires pygame)
cd original-src && python test.py
```

### Documentation
```bash
# View project documentation
cat README.md

# View planning documents
ls materials/
cat materials/new_plan.md
```

### Data Analysis
```bash
# Check available CSV data files (if any)
find . -name "*.csv" -type f

# View CSV file structure (replace with actual file)
head -5 path/to/data.csv
```

## Development Notes

- **No formal testing framework** is currently set up - tests need to be implemented
- **No linting or formatting tools** are configured - consider adding black, flake8, pytest
- **No dependency management** beyond basic pip install
- **Main development files**: `helpers.py`, `ohlcv_feeder.py`
- **Reference implementation**: `original-src/` directory contains Snake DQN tutorial code
- **Planning documents**: `materials/` directory contains detailed future architecture plans