# Task Completion Checklist

## When Development Tasks Are Completed

### Testing Requirements
1. **Run unit tests** to ensure no regressions:
   ```bash
   python -m unittest discover tests/
   ```

2. **Test specific modules** if changes affect core components:
   ```bash
   python -m unittest tests.test_agent  # if agent changes
   python -m unittest tests.test_env    # if environment changes
   python -m unittest tests.test_trainer # if training logic changes
   ```

### No Linting/Formatting Tools
- Project does not have linting tools (flake8, black) configured
- Manual code review for style consistency
- Follow existing code patterns and naming conventions

### Model Validation
- If model architecture changes, verify training still works:
  ```bash
  python run.py --csv sample_data.csv --mode sequential
  ```
- Ensure model saves/loads correctly after changes

### Integration Testing
- If data pipeline changes, test with real CSV data
- If environment changes, verify state representation is still 4 features
- If agent changes, ensure action space remains [0, 1, 2]

### Documentation Updates
- Update CLAUDE.md if architecture or commands change
- Update README.md if core functionality changes
- No automatic documentation generation configured

### No Build Process
- Pure Python project - no build/compilation step required
- Dependencies managed via pip install

### Version Control
- Standard git workflow
- Test before committing changes
- No automated CI/CD pipeline configured

## Critical Tests Before Deployment
1. Verify 4-feature state representation maintained
2. Confirm binary reward system (+10/-10) still works  
3. Test threading synchronization if using threaded mode
4. Validate CSV data loading with required columns
5. Check model convergence on sample data