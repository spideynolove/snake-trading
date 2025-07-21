# V4 Forex DQN: Simplicity Meets Sophistication

## Philosophy: Simplicity Beats Complexity

V4 tests the hypothesis that financial markets behave like sophisticated random walks, where the edge comes not from prediction but from superior **risk management** and **timing**. Inspired by Snake AI, V4 uses minimal input state and trains continuously on each step with immediate feedback.

---

## Core Principles

- **No Technical Indicators**: Use raw market data only  
- **Fixed Position Sizes**: Remove bet sizing as a variable  
- **Immediate Feedback**: Binary profit/loss rewards only  
- **Continuous Learning**: Train every tick like Snake AI  
- **Simple State**: 4 core inputs to represent market state

---

## Minimal State Representation

1. **Price Level**: Normalized within recent range (0–1)  
2. **Volatility**: Standard deviation over last 20–50 candles  
3. **Position Size**: Fixed at 0.01 lots (or % if dynamic sizing enabled)  
4. **Immediate Risk**: Unrealized P&L divided by account balance

---

## Action Space

- **0: Hold** — Maintain current position  
- **1: Buy** — Enter or reverse to long  
- **2: Sell** — Enter or reverse to short

---

## Reward System

- **+10**: Close profitable trade  
- **-10**: Close losing trade  
- **0**: While position is open  
- **Optional**: Intermediate rewards or time-based penalties (via config)

---

## Architecture & Configuration

### Base Model (Simple)
```python
Linear_QNet(4, 256, 3)