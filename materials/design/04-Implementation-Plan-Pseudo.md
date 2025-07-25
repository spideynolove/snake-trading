**Overall System Architecture (Pseudocode)**

```pseudocode
// Local Machine Processes
Initialize DataCrawler(symbols, timeframes)
Initialize SignalGenerator(model_path, symbols)
Initialize RiskManager(initial_balance)
Initialize TelegramSender(bot_token, chat_id, encryption_key)

Loop Forever:
    Await DataCrawler.UpdateAllMarketData()
    Await DataCrawler.UpdateEconomicCalendar()
    Await DataCrawler.UpdateIntermarketData()

    For Each symbol in symbols:
        features = DataCrawler.GetFeaturesForSymbol(symbol)
        If features are valid:
            predicted_action = SignalGenerator.Model.Predict(features)
            confidence = CalculateModelConfidence(predicted_action)
            market_sentiment = EconomicCalendar.GetSentiment()

            If confidence > Threshold AND market_sentiment != "high_volatility":
                adjusted_action = RiskManager.ValidateAndAdjustAction(predicted_action, symbol)
                If adjusted_action is approved:
                    signal = CreateSignalObject(symbol, adjusted_action, calculated_lots, timestamp, confidence, etc.)
                    Await TelegramSender.SendSignal(signal)

    Sleep(Update_Interval_Minutes)


// VPS (Execution Server) Processes
Initialize TelegramReceiver(bot_token, encryption_key)
Initialize ExecutionEngine(mt4_path)

Loop Forever:
    updates = Await TelegramReceiver.GetNewMessages()
    For Each update in updates:
        If update contains "TRADING SIGNAL" and JSON:
            signal_data = ParseSignalFromMessage(update.text)
            If SignalReceiver.ValidateSignal(signal_data): // Checks age, auth, duplicates
                Await ExecutionEngine.ExecuteSignal(signal_data)
    Sleep(Polling_Interval_Seconds)

```

**Component 1: Data Crawling (`data_crawling/`)**

*   **`MarketDataCrawler` (Pseudocode):**
    ```pseudocode
    Class MarketDataCrawler:
        Properties: symbols[], timeframes[], data_cache{}, last_update{}

        Method FetchOandaData(symbol, timeframe, count):
            // Async HTTP GET to OANDA API
            // Process response into standardized DataFrame
            // Return DataFrame or None on error

        Method UpdateAllData():
            // Create async tasks for FetchOandaData for all symbol/timeframe combos
            // Await all tasks
            // Store successful results in data_cache and update last_update

        Method GetFeaturesForSymbol(symbol, primary_timeframe):
            // Retrieve cached data for symbol/timeframe
            // Calculate indicators: pct_change, SMA ratios, RSI, volume ratio
            // Get higher timeframe trend data
            // Return latest feature vector or None

        Method StartRealTimeFeed(update_interval):
            // Loop: Call UpdateAllData(), Sleep(update_interval)

    ```
*   **`EconomicCalendar` (Pseudocode):**
    ```pseudocode
    Class EconomicCalendar:
        Properties: base_url, high_impact_events[]

        Method FetchCalendarData():
            // HTTP GET to calendar API (e.g., ForexFactory)
            // Return parsed JSON or None on error

        Method FilterHighImpactEvents(data, currencies[]):
            // Iterate through fetched events
            // Filter for events within next 24h, High/Medium impact, relevant currencies
            // Return list of filtered event objects

        Method GetMarketSentiment():
            // data = FetchCalendarData()
            // events = FilterHighImpactEvents(data)
            // If many events: return "high_volatility"
            // Elif some events: return "medium_volatility"
            // Else: return "low_volatility"

    ```
*   **`IntermarketData` (Pseudocode - Implied):**
    ```pseudocode
    Class IntermarketDataCrawler: // Similar structure to MarketDataCrawler
        Properties: asset_symbols[], correlation_data_cache{}, last_update{}

        Method FetchAssetData(symbol): // Fetch price, volume for asset (e.g., DXY, Gold)
            // Async HTTP GET to relevant API
            // Process response into standardized data point
            // Return data point or None on error

        Method UpdateAllData():
            // Create async tasks for FetchAssetData for all asset symbols
            // Await all tasks
            // Store results in correlation_data_cache and update last_update

        Method GetCorrelationData(): // Called by SignalGenerator
            // Return correlation_data_cache

        Method StartRealTimeFeed(update_interval):
            // Loop: Call UpdateAllData(), Sleep(update_interval)
    ```

**Component 2: Signal Generation & Communication (Local Machine)**

*   **`SignalGenerator` (Pseudocode):**
    ```pseudocode
    Class SignalGenerator:
        Properties: symbols[], model, data_crawler, economic_cal, risk_manager, positions{}, trade_count, etc.

        Method LoadModel(model_path):
            // Initialize DQN model architecture (matching training)
            // Load weights from model_path
            // Set model to evaluation mode

        Method GenerateSignals():
            // Loop Forever:
                // Reset daily counters if needed
                // Await data_crawler.UpdateAllData() // Market, Calendar, Intermarket
                // market_sentiment = economic_cal.GetMarketSentiment()
                // For Each symbol:
                    // signal = AnalyzeSymbol(symbol, market_sentiment)
                    // If signal: Await SendSignal(signal)
                // Sleep(Analysis_Interval)

        Method AnalyzeSymbol(symbol, market_sentiment):
            // If daily trade limit reached: Return None
            // features = data_crawler.GetFeaturesForSymbol(symbol)
            // If no features: Return None
            // predicted_action, confidence = Model.Predict(features)
            // If position changes AND confidence > threshold AND sentiment allows:
                // lot_size = risk_manager.CalculatePositionSize(symbol, risk_params)
                // signal = CreateSignalObject(...)
                // Update internal position tracking
                // Return signal
            // Else: Return None

        Method SendSignal(signal):
            // telegram_sender = TelegramSender(...)
            // Await telegram_sender.SendSignal(signal)

    ```
*   **`TelegramSender` (Pseudocode):**
    ```pseudocode
    Class TelegramSender:
        Properties: bot_token, chat_id, encryption_key, base_url

        Method GenerateAuthHash(signal_data):
            // If encryption_key:
                // Create data_string from critical signal fields
                // Return HMAC-SHA256 hash (truncated) of data_string
            // Else: Return "no_auth"

        Method SendMessage(text):
            // Async HTTP POST to Telegram Bot API sendMessage endpoint
            // Include chat_id, text, parse_mode

        Method SendSignal(signal_data):
            // Add auth_hash to signal_data
            // Format signal_data as JSON string
            // message = "🤖 TRADING SIGNAL\n```json\n" + JSON_string + "\n```"
            // Await SendMessage(message)

        Method SendStatusUpdate(message):
            // Format timestamped status message
            // Await SendMessage(formatted_message)

        Method SendPerformanceReport(metrics):
            // Format metrics into a report string
            // Await SendMessage(formatted_report)
    ```

**Component 3: VPS Execution Server**

*   **`TelegramReceiver` (Pseudocode):**
    ```pseudocode
    Class TelegramReceiver:
        Properties: bot_token, encryption_key, base_url, last_update_id, executor, processed_signals{}

        Method GetUpdates():
            // Async HTTP GET to Telegram getUpdates endpoint
            // Use offset=last_update_id + 1
            // Update last_update_id from response
            // Return list of updates

        Method ProcessUpdate(update):
            // If update contains 'message' and 'text':
                // text = update.message.text
                // If text contains "TRADING SIGNAL" and "```json":
                    // Extract JSON string from text
                    // signal_data = Parse JSON string
                    // Await ProcessSignal(signal_data)

        Method ProcessSignal(signal_data):
            // Validate signal structure (required fields present)
            // Check signal age (reject if too old)
            // Verify auth_hash (using encryption_key)
            // Check for duplicates (using signal_id)
            // If all checks pass:
                // Await executor.ExecuteSignal(signal_data)
                // Add signal_id to processed_signals
                // Cleanup old processed_signals

        Method StartListening():
            // Loop Forever:
                // updates = Await GetUpdates()
                // For Each update: Await ProcessUpdate(update)
                // Sleep(Polling_Interval)

    ```
*   **`ExecutionEngine` (Pseudocode):**
    ```pseudocode
    Class ExecutionEngine: // Uses DWXConnect library
        Properties: mt4_files_path, dwx_client, risk_limits{}, execution_log[]

        Method ExecuteSignal(signal_data):
            // Extract symbol, order_type, lots from signal_data
            // Apply VPS-side safety limits (max_lot_size, etc.)
            // Validate symbol is allowed
            // Check existing positions for symbol
            // If limits not exceeded:
                // If order_type == "buy": ExecuteBuy(...)
                // Elif order_type == "sell": ExecuteSell(...)
                // Elif order_type == "close": ExecuteClose(...)
            // Log execution result

        Method ExecuteBuy/Sell(symbol, lots, signal_data):
            // Get current market price (bid/ask) via dwx_client
            // Calculate SL/TP based on price and signal_data or defaults
            // Call dwx_client.open_order(...) with parameters
            // Log execution

        Method ExecuteClose(symbol, signal_data):
            // Iterate through dwx_client.open_orders
            // Find orders matching symbol
            // Call dwx_client.close_order(order_id) for each
            // Log execution

        Method CalculateSL/TP(entry_price, direction, signal_data):
            // Extract SL/TP pips or values from signal_data if provided
            // Else, use default fixed pip values
            // Calculate price levels based on entry_price and direction
            // Return calculated SL/TP prices

        Method LogExecution(signal_data, success, error):
            // Create log entry with timestamp, signal, success status, error
            // Append to execution_log
            // Maintain log size limit

    ```

**Component 4: Risk Management & Monitoring (Local Machine)**

*   **`RiskManager` (Pseudocode):**
    ```pseudocode
    Class RiskManager:
        Properties: balance, risk_limits{}, open_positions{}, trade_history[], daily_stats{}

        Method ValidateSignal(signal_data):
            // Reset daily counters if new day
            // Check: daily trade count < limit
            // Check: daily P&L > -max_daily_loss_percent * initial_balance
            // Check: current drawdown < max_drawdown_percent
            // Check: signal lots <= CalculateMaxPositionSize()
            // Return (True, "Approved") OR (False, "Rejection Reason")

        Method CalculateMaxPositionSize():
            // Based on current balance, max_risk_per_trade, stop_loss_pips, pip_value
            // Calculate maximum allowable lots
            // Return min(calculated_lots, absolute_max_lots)

        Method UpdatePosition(symbol, action, lots, price):
            // Record new position details (symbol, action, lots, entry_price, timestamp)
            // Update internal tracking (open_positions, daily_trade_count)

        Method CalculateUnrealizedPnL(symbol, current_price):
            // Retrieve position details for symbol
            // Calculate P&L based on entry price, current price, action, lots
            // Return P&L amount

        Method ClosePosition(symbol, exit_price):
            // Calculate realized P&L using CalculateUnrealizedPnL
            // Update balance, daily P&L, peak balance
            // Log trade details (timestamp, symbol, action, lots, entry/exit prices, P&L, final balance)
            // Remove position from open_positions

        Method GetRiskMetrics():
            // Calculate current balance, total return, drawdown, daily P&L, trade count, win rate, etc.
            // Return metrics dictionary

        Method EmergencyStop():
            // Log emergency stop trigger and reason
            // Return stop command/metrics (could be used to halt signal generation)
    ```