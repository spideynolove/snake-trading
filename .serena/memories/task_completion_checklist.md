# Snake Trading AI - Task Completion Checklist

## When a coding task is completed, perform these steps:

### 1. Code Quality Checks

#### Manual Review
- [ ] **Code style consistency**: Follow established naming conventions (snake_case, PascalCase)
- [ ] **Type hints**: Add type hints for new functions, especially those handling DataFrames
- [ ] **Error handling**: Include appropriate try-except blocks for file operations and data parsing
- [ ] **Documentation**: Add docstrings for new classes and complex methods

#### No Automated Tools (Yet)
- **No linting configured** - Consider manually checking for PEP 8 compliance
- **No formatting tool** - Maintain consistent spacing and indentation manually
- **No type checker** - Manually verify type hints are correct

### 2. Testing Strategy

#### Current State: No Test Framework
- **Manual testing only** - Run the code with sample data to verify functionality
- **Test data processing**: Use `python ohlcv_feeder.py` to test OHLCV functionality
- **Test imports**: Verify new modules can be imported without errors

#### Future Testing (When Implemented)
- [ ] **Unit tests**: Write tests for individual functions/methods
- [ ] **Integration tests**: Test data processing pipelines end-to-end
- [ ] **Regression tests**: Ensure changes don't break existing functionality

### 3. Documentation Updates

#### Required Updates
- [ ] **Update CLAUDE.md** if architecture changes significantly
- [ ] **Update README.md** if new features are added or commands change
- [ ] **Update docstrings** for modified methods
- [ ] **Comment complex algorithms** especially in financial calculations

#### Planning Documents
- [ ] **Check materials/new_plan.md** for alignment with future architecture
- [ ] **Update planning docs** if implementation deviates from plan

### 4. Data Validation

#### For Data Processing Changes
- [ ] **Test with sample CSV**: Ensure OHLCV data processing still works
- [ ] **Validate output format**: Check that pandas DataFrame outputs are correct
- [ ] **Performance check**: Verify processing speed hasn't degraded significantly
- [ ] **Memory usage**: Monitor for potential memory leaks in data processing

### 5. Integration Checks

#### Module Dependencies
- [ ] **Import validation**: Ensure all imports work correctly
- [ ] **Backward compatibility**: Existing functionality should continue working
- [ ] **Cross-module compatibility**: New changes shouldn't break other modules

### 6. Version Control

#### Git Workflow
- [ ] **Stage changes**: `git add .` for modified files
- [ ] **Check git status**: Verify correct files are staged
- [ ] **Commit with descriptive message**: Follow established commit message patterns
- [ ] **Review diff before commit**: `git diff --cached` to see what's being committed

### 7. Future Development Readiness

#### DQN Implementation Preparation
- [ ] **PyTorch compatibility**: Ensure changes won't conflict with future torch implementation
- [ ] **Data pipeline ready**: Verify data processing can feed into DQN training
- [ ] **Configuration flexibility**: Ensure new features are configurable

### 8. Performance Considerations

#### For Data Processing Tasks
- [ ] **Memory efficiency**: Use `.copy()` pattern for DataFrame operations
- [ ] **Computational efficiency**: Avoid unnecessary loops in pandas operations
- [ ] **Threading safety**: Consider thread safety if modifying ohlcv_feeder.py

### 9. Specific Checks by Module

#### helpers.py Changes
- [ ] **Test technical indicators**: Verify TA-Lib functions work correctly
- [ ] **Validate calculations**: Double-check financial calculations and formulas
- [ ] **Configuration loading**: Ensure config files load properly

#### ohlcv_feeder.py Changes
- [ ] **CSV parsing**: Test with different CSV formats and delimiters
- [ ] **Threading behavior**: Verify background threading works correctly
- [ ] **Memory management**: Check for data accumulation issues

#### New Module Development
- [ ] **Follow existing patterns**: Use established class and method structure
- [ ] **Import organization**: Follow standard library → third-party → local pattern
- [ ] **Configuration support**: Add configuration dictionary support where appropriate

## Emergency Rollback Procedure

If issues are discovered after implementation:

1. **Immediate**: `git status` to see current state
2. **Review**: `git diff` to see recent changes
3. **Revert**: `git checkout -- <file>` to revert specific files
4. **Alternative**: `git reset --hard HEAD~1` to revert last commit (use cautiously)

## Future Automation Goals

When the project matures, consider implementing:
- **pytest** for automated testing
- **black** for code formatting
- **flake8** or **pylint** for linting
- **mypy** for type checking
- **pre-commit hooks** for automated checks