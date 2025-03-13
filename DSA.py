import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Assume your existing modules are available:
from technical_analysis import analyze_stock
from data_visualisations import plot_or_save_stock_data, download_results_as_pdf

###############################################
# Helper Function: Check Signal Validity       #
###############################################

def is_valid_signal(test_data, validity_window_days):
    """
    Checks if the latest signal timestamp is within the validity window.
    Assumes test_data.index contains datetime objects.
    """
    last_date = test_data.index[-1]
    if not isinstance(last_date, datetime):
        last_date = pd.to_datetime(last_date)
    return (datetime.now() - last_date) <= timedelta(days=validity_window_days)

###############################################
# Compute Effective Score for a Candidate Stock #
###############################################

def compute_effective_score(result, max_investment):
    """
    Computes an effective score for a candidate stock using today's price and potential upside.
    
    Calculation details:
      - potential_upside is assumed to be computed as: ((Target Price - Current Price) / Current Price * 100).
      - Trading Signal: if the most recent valid signal is a buy, add +10; if sell, subtract 10; if neutral, 0.
      - Risk Adjustment: if RSI < 30 (undervalued) add +5, if RSI > 70 (overbought) subtract 5.
    
    The effective score (in percentage points) is then converted to an effective dollar return when investing max_investment dollars.
    
    Returns:
      cost (which is max_investment),
      effective_value in dollars,
      and a details dictionary with key metrics.
    """
    current_price = result.get("Current Price ($)")
    target_price = result.get("Target Price ($)")
    potential_upside = result.get("Potential Upside (%)")  # e.g. 15 means 15%
    test_data = result.get("Test Data")
    
    # Ensure test_data index is datetime for signal validation
    if test_data is not None and "Date" in test_data.columns:
        test_data.index = pd.to_datetime(test_data["Date"])
    
    # Determine the trading signal (for simplicity, use the latest row):
    signal = "Neutral"
    if test_data is not None and not test_data.empty:
        latest_row = test_data.iloc[-1]
        if latest_row.get("Buy_Signal"):
            signal = "Buy"
        elif latest_row.get("Sell_Signal"):
            signal = "Sell"
    
    # Signal bonus: +10 for Buy, -10 for Sell
    signal_bonus = 10 if signal == "Buy" else (-10 if signal == "Sell" else 0)
    
    # Risk bonus: +5 if RSI < 30, -5 if RSI > 70, else 0
    rsi = None
    if test_data is not None and not test_data.empty:
        rsi = test_data.iloc[-1].get("RSI")
    risk_bonus = 0
    if rsi is not None:
        if rsi < 30:
            risk_bonus = 5
        elif rsi > 70:
            risk_bonus = -5

    # Effective score (in percentage points)
    effective_percentage = (potential_upside if potential_upside is not None else 0) + signal_bonus + risk_bonus

    # Effective dollar return if investing max_investment dollars:
    effective_value = max_investment * effective_percentage / 100

    details = {
        "Current Price": current_price,
        "Target Price": target_price,
        "Potential Upside (%)": potential_upside,
        "RSI": rsi,
        "Trading Signal": signal,
        "Effective Score (%)": effective_percentage,
        "Effective Return ($)": effective_value
    }
    return max_investment, effective_value, details

###############################################
# Dynamic Programming Knapsack Selection      #
###############################################

def portfolio_dp(candidates, capacity_slots):
    """
    Selects a subset of candidate stocks using a 0/1 knapsack DP approach.
    
    Each candidate is a tuple: (ticker, cost, effective_value, details)
    Here, cost is fixed (i.e. max investment per stock) and capacity_slots is the number of stocks 
    that can be purchased (budget // max_investment).
    
    Since each candidate costs 1 slot, the optimal solution is to pick the candidates with the highest 
    effective_value until the capacity is filled.
    
    Returns:
      total_effective_value and a list of selected candidate tuples.
    """
    # Sort candidates by effective_value descending and pick the top capacity_slots.
    sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    selected = sorted_candidates[:capacity_slots]
    total_effective_value = sum([cand[2] for cand in selected])
    return total_effective_value, selected

###############################################
# Real-Time Risk Dashboard (Using Dictionary) #
###############################################

def risk_dashboard(all_tickers):
    st.subheader("Real-Time Risk Dashboard")
    risk_table = {}
    for ticker in all_tickers:
        result = analyze_stock(ticker)
        test_data = result.get("Test Data")
        if test_data is not None and not test_data.empty:
            if "Date" in test_data.columns:
                test_data.index = pd.to_datetime(test_data["Date"])
            latest_data = test_data.iloc[-1]
            rsi = latest_data.get("RSI")
            macd = latest_data.get("MACD")
            risk_table[ticker] = {
                "RSI": rsi,
                "MACD": macd,
                "Current Price": result.get("Current Price ($)")
            }
    if not risk_table:
        st.write("No risk data available.")
        return
    risk_df = pd.DataFrame.from_dict(risk_table, orient="index")
    risk_df.index.name = "Ticker"
    st.dataframe(risk_df)
    st.caption("Low RSI (<30) suggests undervaluation; high RSI (>70) suggests overvaluation.")

###############################################
# Main Recommendation Page                    #
###############################################

def recommendation_page(all_tickers):
    st.header("Stock Recommendation Engine")
    st.write("""
    Our system automatically recommends stocks to invest in. Given a total budget and a maximum 
    investment per stock, we select the best candidates based on today's price, potential upside, 
    recent trading signals, and risk (RSI). Overpriced stocks with recent sell signals are penalized, 
    while undervalued stocks with recent buy signals are favored.
    """)
    
    # Default parameters:
    total_budget = st.number_input("Total Investment Budget ($)", min_value=1000, value=20000, step=1000)
    max_per_stock = st.number_input("Maximum Investment per Stock ($)", min_value=100, value=10000, step=1000)
    validity_window = st.slider("Signal Validity Window (days)", min_value=1, max_value=30, value=7)
    
    # Calculate number of stocks (slots) that can be purchased:
    capacity_slots = total_budget // max_per_stock
    st.write(f"Based on your budget, our algorithm will recommend up to **{capacity_slots} stocks**.")
    
    # Process each stock from the predetermined list:
    candidates = []
    candidate_details_list = []
    for ticker in all_tickers:
        result = analyze_stock(ticker)
        test_data = result.get("Test Data")
        if test_data is not None and not test_data.empty:
            # Ensure signal dates are properly formatted if available.
            if "Date" in test_data.columns:
                test_data.index = pd.to_datetime(test_data["Date"])
            if is_valid_signal(test_data, validity_window):
                cost, eff_value, details = compute_effective_score(result, max_per_stock)
                # Only consider stocks with a positive effective return.
                if eff_value > 0:
                    candidates.append((ticker, cost, eff_value, details))
                    candidate_details_list.append({
                        "Ticker": ticker,
                        "Current Price": details["Current Price"],
                        "Target Price": details["Target Price"],
                        "Potential Upside (%)": details["Potential Upside (%)"],
                        "RSI": details["RSI"],
                        "Trading Signal": details["Trading Signal"],
                        "Effective Score (%)": details["Effective Score (%)"],
                        "Effective Return ($)": round(eff_value, 2)
                    })
    
    if not candidates:
        st.write("No candidate stocks with strong valid buy signals were found.")
        return

    # --- Display the Recommended Portfolio First ---
    total_eff_val, recommended = portfolio_dp(candidates, capacity_slots)
    
    st.markdown("## Recommended Portfolio")
    st.markdown(f"Total Expected Return: ${total_eff_val:,.2f} when investing ${max_per_stock:,.2f} in each stock.")
    
    recommended_list = []
    for item in recommended:
        ticker, cost, eff_value, details = item
        recommended_list.append({
            "Ticker": ticker,
            "Current Price": details["Current Price"],
            "Target Price": details["Target Price"],
            "Potential Upside (%)": details["Potential Upside (%)"],
            "RSI": details["RSI"],
            "Trading Signal": details["Trading Signal"],
            "Effective Score (%)": details["Effective Score (%)"],
            "Effective Return ($)": round(eff_value, 2)
        })
    df_recommended = pd.DataFrame(recommended_list)
    st.dataframe(df_recommended)

    # --- Then Display Candidate Stock Details ---
    st.markdown("### Candidate Stock Details")
    df_candidates = pd.DataFrame(candidate_details_list)
    st.dataframe(df_candidates)
    
    # --- Detailed Analysis of Recommended Stocks ---
    st.markdown("### Detailed Analysis of Recommended Stocks")
    for item in recommended:
        ticker, cost, eff_value, details = item
        st.markdown(f"#### {ticker}")
        st.write(f"**Current Price (today):** ${details['Current Price']}")
        st.write(f"**Target Price:** ${details['Target Price']}")
        st.write(f"**Potential Upside:** {details['Potential Upside (%)']}%")
        st.write(f"**RSI:** {details['RSI']}")
        st.write(f"**Trading Signal:** {details['Trading Signal']}")
        st.write(f"**Effective Return:** ${round(eff_value, 2)} based on an effective score of {details['Effective Score (%)']}%")
        st.caption("This stock was selected because it shows a strong, valid buy signal, favorable potential upside, and low risk (RSI).")
        st.write("**Chart:**")
        # Display chart using your plotting function
        plot_or_save_stock_data(ticker, result.get("Test Data"), action='plot')
    
    st.write("---")
    risk_dashboard(all_tickers)

###############################################
# Algorithm Back Testing Page                 #
###############################################

def back_testing_page(all_tickers):
    st.title("Algorithm Back Testing")
    st.write("""
    This page shows the historical performance of our algorithm.
    Here we backtest our trading strategy and detailed analysis of stocks using historical data.
    """)
    
    # (You can integrate your existing backtesting code here.)
    # For illustration, we display the detailed analysis table for a selection of stocks.
    selected_tickers = st.multiselect("Select Tickers for Back Testing", options=all_tickers)
    if selected_tickers:
        results = []
        images = []
        for ticker in selected_tickers:
            result = analyze_stock(ticker)
            results.append(result)
        df_results = pd.DataFrame(results)
        st.markdown("### Back Testing Results")
        st.dataframe(df_results[['Ticker', 'Current Price ($)', 'Target Price ($)', 'Potential Upside (%)', 
                                   'Trades Closed', 'Average Return per Trade (%)', 'Total Return (%)']])
        st.write("### Historical Stock Charts")
        for ticker in selected_tickers:
            st.write(f"#### {ticker}")
            plot_or_save_stock_data(ticker, analyze_stock(ticker).get("Test Data"), action='plot')

###############################################
# Main Program with Top-Bar Navigation (Tabs) #
###############################################

def main():
    # Predefined list of stocks (your universe for recommendation)
    all_tickers = [
        'AAPL', 'META', 'AMZN', 'NVDA', 'MSFT', 'NIO', 'XPEV', 'LI', 'ZK',
        'PYPL', 'AXP', 'MA', 'GPN', 'V', 'FUTU',
        'GS', 'JPM', 'BLK', 'C', 'BX',
        'KO', 'WMT', 'MCD', 'NKE', 'SBUX'
    ]
    
    # Create top-bar navigation using tabs
    tabs = st.tabs(["Stock Recommendation", "Algorithm Back Testing"])
    
    with tabs[0]:
        recommendation_page(all_tickers)
        
    with tabs[1]:
        back_testing_page(all_tickers)

if __name__ == "__main__":
    main()
