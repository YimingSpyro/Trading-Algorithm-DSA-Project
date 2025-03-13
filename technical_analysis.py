import yfinance as yf
from technical_indicators import calculate_indicators, define_signals
from data_preprocessing import clean, get_analyst_ratings

# Global cache (hash map) to store analysis results per ticker.
# This allows us to reuse computed information when user preferences change.
analysis_cache = {}

###############################################
# Helper Function: Calculate Average Sell Signals
###############################################
def average_sell_signals(data):
    sell_count = 0
    buy_count = 0
    buy_state = 0
    total_sell_signals_between_buys = 0

    for i in range(len(data)):
        if data['Buy_Signal'].iloc[i]:
            buy_state += 1
            if buy_state >= 1:
                total_sell_signals_between_buys += sell_count
                sell_count = 0  
        elif data['Sell_Signal'].iloc[i]:
            if buy_state >= 1:
                buy_state = 0
                buy_count += 1
            sell_count += 1

    average = total_sell_signals_between_buys / buy_count if buy_count > 0 else 0
    return average

###############################################
# Trading Strategy Function
###############################################
def trading_strategy(data, avg_sell_signals, initial_amount=10000):
    balance = initial_amount
    stock_quantity = 0
    profits = []
    in_trade = False
    buy_price = None
    sell_signal_count = 0

    # Define a threshold based on the average sell signals
    sell_threshold = avg_sell_signals / 2

    for i in range(len(data)):
        if data['Buy_Signal'].iloc[i] and not in_trade:
            stock_quantity = balance / data['Close'].iloc[i]
            balance = 0
            buy_price = data['Close'].iloc[i]
            in_trade = True
            sell_signal_count = 0 
        elif data['Sell_Signal'].iloc[i] and in_trade:
            sell_signal_count += 1
            if sell_signal_count >= sell_threshold:
                balance = stock_quantity * data['Close'].iloc[i]
                stock_quantity = 0
                sell_price = data['Close'].iloc[i]
                profit_percent = ((sell_price - buy_price) / buy_price) * 100
                profits.append(profit_percent)
                in_trade = False
                sell_signal_count = 0

    # If still in trade at the end, use the last available closing price.
    if in_trade:
        balance = stock_quantity * data['Close'].iloc[-1]
    
    final_amount = balance
    return profits, final_amount, in_trade  # Return in_trade status

###############################################
# Performance Calculation Function
###############################################
def calculate_performance(profits, initial_amount, final_amount, period_years):
    total_trades = len(profits)
    avg_return_per_trade = sum(profits) / total_trades if total_trades > 0 else 0
    total_return = ((final_amount - initial_amount) / initial_amount) * 100
    avg_annual_return = ((final_amount / initial_amount) ** (1 / period_years) - 1) * 100 if period_years > 0 else 0
    return total_trades, avg_return_per_trade, total_return, avg_annual_return

###############################################
# Upside Calculation Function
###############################################
def calculate_upside(current_price, target_price):
    if target_price <= 0:
        raise ValueError("Target price must be positive")
    return ((target_price - current_price) / current_price) * 100

###############################################
# Main Analysis Function (with caching for dynamic programming)
###############################################
def analyze_stock(ticker):
    # If the analysis for this ticker has already been computed, return it.
    if ticker in analysis_cache:
        return analysis_cache[ticker]
    
    # Download and clean data (10 years period)
    data = clean(yf.download(ticker, period='10y'))
    
    # Check if data was downloaded successfully and is sufficient
    if data.empty or len(data) < 252:
        result = {
            'Ticker': ticker,
            'Current Price ($)': None,
            'Target Price ($)': None,
            'Potential Upside (%)': None,
            'Trades Closed': 0,
            'Average Return per Trade (%)': None,
            'Total Return (%)': None,
            'In Trade': None,
            'Train Data': data,
            'Test Data': data
        }
        analysis_cache[ticker] = result
        return result
    
    # Split data into training and testing sets
    train_data = data[:-252]
    test_data = data[-252:]
    
    # Create copies to avoid modifying original data
    train_data = train_data.copy()
    test_data = test_data.copy()

    # Calculate technical indicators for both sets
    train_data['MACD'], train_data['Signal'], train_data['RSI'], train_data['Upper_Band'], train_data['Lower_Band'] = calculate_indicators(train_data)
    test_data['MACD'], test_data['Signal'], test_data['RSI'], test_data['Upper_Band'], test_data['Lower_Band'] = calculate_indicators(test_data)

    # Define trading signals
    train_data = define_signals(train_data)
    test_data = define_signals(test_data)

    # Compute average sell signals for train and test data
    avg_sell_signals_train = average_sell_signals(train_data)
    avg_sell_signals_test = average_sell_signals(test_data)

    # Run the trading strategy on the test data
    test_profits, test_final_balance, in_trade = trading_strategy(test_data, avg_sell_signals_test)

    # Calculate performance metrics for the test period
    test_years = len(test_data) / 252
    test_total_trades, test_avg_return_per_trade, test_total_return, test_avg_annual_return = calculate_performance(
        test_profits, 10000, test_final_balance, test_years)

    # Retrieve analyst ratings and target price
    info = get_analyst_ratings(ticker)
    target_price = info.get('targetMeanPrice')

    # Current price is now the last closing price of the entire dataset.
    if data.empty or 'Close' not in data.columns or data['Close'].empty:
        current_price = None
    else:
        current_price = data['Close'].iloc[-1]

    # Calculate potential upside only if both current and target prices are available
    if current_price is None or target_price is None:
        upside = None
    else:
        try:
            upside = calculate_upside(current_price, target_price)
        except ValueError:
            upside = None

    result = {
        'Ticker': ticker,
        'Current Price ($)': round(current_price, 2) if current_price is not None else None,
        'Target Price ($)': round(target_price, 2) if target_price is not None else None,
        'Potential Upside (%)': round(upside, 2) if upside is not None else None,
        'Trades Closed': test_total_trades,
        'Average Return per Trade (%)': round(test_avg_return_per_trade, 2) if test_avg_return_per_trade is not None else None,
        'Total Return (%)': round(test_total_return, 2) if test_total_return is not None else None,
        'In Trade': in_trade,
        'Train Data': train_data,
        'Test Data': test_data
    }
    
    # Store the computed result in the cache for future use.
    analysis_cache[ticker] = result
    return result
