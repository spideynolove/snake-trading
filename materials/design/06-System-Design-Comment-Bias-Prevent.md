## Look Ahead Bias pseudocode

```
// --- Shared Resources (Thread-Safe) ---
DATA_QUEUE = new Queue()         // Holds incoming data points (bars/ticks)
ORDERS_QUEUE = new Queue()       // Holds generated orders
SYNCHRONIZATION_EVENTS = {
    F1: new Event(),             // Signal: Execution/Done -> Data Feed
    F2: new Event(),             // Signal: Data Feed -> Trading Logic
    F3: new Event()              // Signal: Trading Logic -> Execution
}

// --- Thread 1: Data Feeding ---
FUNCTION DataFeedingThread():
    WHILE (data_source_has_more_data):
        // 1. Read the *next* piece of data (e.g., one bar)
        NEW_DATA_POINT = ReadNextDataPoint()

        // 2. Put it in the queue for the logic thread
        DATA_QUEUE.Put(NEW_DATA_POINT)

        // 3. Signal the Trading Logic thread that new data is ready
        SYNCHRONIZATION_EVENTS.F2.Set()

        // 4. Wait for the Trading Logic and Execution cycle to complete
        //    This is the key step preventing look-ahead:
        //    The data thread STOPS here until the logic for *this* data point is done.
        SYNCHRONIZATION_EVENTS.F1.Wait()

        // 5. Clear the event for the next cycle
        SYNCHRONIZATION_EVENTS.F1.Clear()

// --- Thread 2: Trading Logic ---
FUNCTION TradingLogicThread(RISK_MANAGER):
    WHILE (True):
        // 1. Wait for the signal that new data is available
        //    This ensures logic only runs when data is ready.
        SYNCHRONIZATION_EVENTS.F2.Wait()

        // 2. Retrieve the *latest* data point from the queue
        //    This is the CURRENT data point for decision making.
        CURRENT_DATA = DATA_QUEUE.Get()

        // 3. --- CRITICAL: Make decisions based ONLY on CURRENT_DATA and prior state ---
        //     - Calculate indicators using CURRENT_DATA and historical data up to this point.
        //     - Apply trading rules.
        //     - Check risk management rules (using state updated by Execution Thread).
        ACTION = DetermineAction(CURRENT_DATA, RISK_MANAGER.GetState())

        // 4. If an action is decided, put the order in the queue
        IF (ACTION is not None):
            NEW_ORDER = CreateOrder(ACTION, CURRENT_DATA)
            ORDERS_QUEUE.Put(NEW_ORDER)

        // 5. Signal the Execution thread that orders are ready to process
        SYNCHRONIZATION_EVENTS.F3.Set()

        // 6. Wait for the Execution thread to finish processing orders
        //    Ensures state is updated before the next data point is fed.
        SYNCHRONIZATION_EVENTS.F1.Wait()

        // 7. Clear the event for the next cycle
        SYNCHRONIZATION_EVENTS.F2.Clear()

// --- Thread 3: Order Execution ---
FUNCTION ExecutionThread(RISK_MANAGER):
    WHILE (True):
        // 1. Wait for the signal that orders are ready
        SYNCHRONIZATION_EVENTS.F3.Wait()

        // 2. Process all pending orders in the queue
        WHILE (not ORDERS_QUEUE.IsEmpty()):
            ORDER = ORDERS_QUEUE.Get()
            // Execute order based on CURRENT_DATA (e.g., market price of the bar)
            EXECUTION_RESULT = ExecuteOrder(ORDER)

            // 3. Update system state (position, equity, risk metrics)
            //     This state is read by the Trading Logic thread.
            RISK_MANAGER.UpdateState(EXECUTION_RESULT)

        // 4. Signal back to the Data Feeding thread that execution is complete
        //     This allows the data feeding loop to proceed to the *next* data point.
        SYNCHRONIZATION_EVENTS.F1.Set()

        // 5. Clear the event for the next cycle
        SYNCHRONIZATION_EVENTS.F3.Clear()

// --- Main Execution ---
MAIN():
    START DataFeedingThread()
    START TradingLogicThread(RISK_MANAGER_INSTANCE)
    START ExecutionThread(RISK_MANAGER_INSTANCE)

    JOIN All Threads // Wait for completion (e.g., end of backtest data)
```

---

## **Analysis of 03-Original-System-Design.md**

### **✅ Usable Components**

**Bridge Architecture Concept**
- Clean separation between training and live execution
- Minimal modifications to existing codebases
- State consistency between gym and live environments

**Action Translation Logic**
- Direct mapping: gym actions → DWX commands
- Simple position state tracking
- Clear execution flow

**Feature Engineering Bridge**
- Consistent state calculation between training/live
- Replicable technical indicator computation
- Standardized observation format

### **❌ Not Usable Components**

**Complex State Harmonization**
- Over-engineered feature matching between gym and DWX
- Too many technical indicators for V12 simplicity goal
- Contradicts 4-feature minimal state approach

**Multiple Position States**
- `[-1, 0, 1]` positions add Hold complexity
- Should be binary Long/Short for V12
- Flat position creates action space complexity

**Heavy Model Integration**
- Loading full DQN models in live bridge
- Real-time inference overhead
- Should start with binary classifier

## **Analysis of Look-Ahead Prevention Pseudocode**

### **✅ Excellent for Real Trading Simulation**

**Thread Synchronization Pattern**
- Perfect for preventing future data access
- Mimics real-time constraints accurately
- Forces sequential processing like Snake game

**Event-Driven Architecture**
- `F1`, `F2`, `F3` events ensure temporal ordering
- No component can "peek ahead" at future bars
- Maintains strict chronological execution

**Queue-Based Data Flow**
- Single data point processing
- Forced waiting between decisions
- Realistic trading latency simulation

### **Critical for V12 Implementation**

**Why This Matters for Snake-like Environment:**
- Snake game: agent sees only current state, makes immediate decision
- Forex trading: must see only current bar, make immediate decision
- Both require strict temporal constraints

**Prevents Common RL Training Errors:**
- No accidentally using future OHLC data for current decisions
- No lookahead bias in feature calculation
- Forces agent to learn from realistic information flow

## **Recommended Hybrid Implementation**

### **From 03-Original-System-Design.md - Keep:**
- Bridge module concept
- Action translation framework
- DWX integration approach

### **From Look-Ahead Prevention - Keep:**
- Complete thread synchronization system
- Event-driven data flow
- Queue-based processing

### **Modified for V12:**

**Simplified Bridge:**
```pseudocode
Class SimpleBridge:
    Method ConvertToGymState(current_bar):
        // Only 4 features
        momentum = calculate_momentum(current_bar)
        position = get_current_position()
        unrealized_pnl = calculate_pnl(current_bar)
        session_time = get_session_indicator(current_bar)
        Return [momentum, position, unrealized_pnl, session_time]
    
    Method ExecuteAction(binary_prediction):
        If prediction > 0.5: DWX.open_long()
        Else: DWX.open_short()
```

**Thread Implementation for V12:**
```pseudocode
// DataFeedingThread: Feed H1 bars sequentially
// TradingLogicThread: Binary prediction on 4 features
// ExecutionThread: Long/Short execution via DWX
```

**This combination gives you:**
- Snake-like real-time constraints
- Minimal complexity (4 features, binary actions)
- Proven DWX execution framework
- Zero lookahead bias
- Realistic trading simulation

The look-ahead prevention pseudocode is **essential** for creating a proper Snake-like trading environment where the agent truly learns under realistic constraints.