"""
main.py

This script is the main entry point for the Streamlit application that performs stock analysis 
and trading strategy evaluation. The application allows users to select stocks, analyze their 
performance, and review the results based on technical indicators and trading strategies. The 
results include stock ranking, trade bot performance, and visualizations of key indicators.

Features:
    - Allows the user to select stocks from a predefined list of tickers for analysis.
    - Analyzes selected stocks based on historical data using technical analysis functions.
    - Displays stock ranking and trading performance metrics, such as potential upside, 
      total return, and average return per trade.
    - Provides graphical visualizations of stock data, including key indicators such as 
      Bollinger Bands, MACD, and RSI, with buy and sell signals.
    - Optionally allows the user to download the analysis results as a PDF report (commented out in the code).

Modules:
    - streamlit: Used for creating the interactive web interface.
    - pandas: Utilized for data manipulation and organizing the analysis results.
    - technical_analysis: Contains the 'analyze_stock' function for performing the stock analysis.
    - data_visualizations: Provides 'plot_or_save_stock_data' for plotting stock data and 
      'download_results_as_pdf' for generating the PDF report.

Usage:
    Run the script to start the Streamlit application. The user can interact with the web interface 
    to select stocks, view analysis results, and visualize data. The application can be extended to 
    include additional functionalities, such as downloading a PDF report.
"""

import streamlit as st
import pandas as pd 
from technical_analysis import analyze_stock
from data_visualisations import plot_or_save_stock_data, download_results_as_pdf

# List of stocks to analyze
tickers = [
    # Technology
    'AAPL', 'META', 'AMZN', 'NVDA', 'MSFT', 'NIO', 'XPEV', 'LI', 'ZK',
    
    # FinTech
    'PYPL', 'AXP', 'MA', 'GPN', 'V', 'FUTU',
    
    # Finance
    'GS', 'JPM', 'BLK', 'C', 'BX',
    
    # Consumer
    'KO', 'WMT', 'MCD', 'NKE', 'SBUX'
]

# Main Streamlit Program
def main():
    st.title("Stock Analysis and Trading Strategy📉📈")
    st.write("1. Select the stocks you want to analyse.")
    st.write("2. Review our analysis and trading strategy performance.")
    st.write("3. Save as pdf for future reference.")
    
    # Create a checkbox for each ticker to analyze
    selected_tickers = st.multiselect("Select Tickers to Analyze", tickers)
    
    if selected_tickers:
        results = []
        top_stock_images = []  # Store paths for top stock images
        for ticker in selected_tickers:
            result = analyze_stock(ticker)
            results.append(result)

        # Stock Ranking Section
        results_df = pd.DataFrame(results)
        stock_ranking_df = results_df.sort_values(by='Potential Upside (%)', ascending=False).copy()
        stock_ranking_df['Rank'] = results_df['Potential Upside (%)'].rank(ascending=False) 
        trade_bot_df = results_df.sort_values(by='Total Return (%)', ascending=False).copy()
        trade_bot_df['Rank'] = results_df['Total Return (%)'].rank(ascending=False) 
        

        st.write("### Stock Ranking:")
        st.write("*Current Price refers to the last closing price of training dataset.")
        st.write("*Target Price refers to the mean analyst price target.")
        st.dataframe(stock_ranking_df[['Rank', 'Ticker', 'Current Price ($)', 'Target Price ($)', 'Potential Upside (%)']])

        # Trade Bot Performance Section
        st.write("### Trade Bot Performance:")
        st.write("*Trading performance based on 1 year of test data.")
        performance_df = trade_bot_df[['Rank','Ticker', 'Trades Closed', 'Average Return per Trade (%)', 'Total Return (%)', 'In Trade']].copy()
        performance_df.rename(columns={'In Trade': 'Status'}, inplace=True)
        performance_df['Status'] = performance_df['Status'].apply(lambda x: "In Trade" if x else "Closed")  # Set Status
        st.dataframe(performance_df)

        # Graph Section for Top 3 Stocks
        st.write("### Analysis of Selected Stocks: Ordered by Potential Return")
        top_stocks = trade_bot_df.head()
        for index, row in top_stocks.iterrows():
            st.write(f"{row['Ticker']} Stock Data")
            plot_or_save_stock_data(row['Ticker'], row['Test Data'], action='plot')

        # Generate images for stocks selected
        top_stocks = trade_bot_df.head()
        for index, row in top_stocks.iterrows():
            image_path = plot_or_save_stock_data(row['Ticker'], row['Test Data'], action='save')
            top_stock_images.append(image_path)

        # Download PDF button
        pdf_file_path = download_results_as_pdf(stock_ranking_df, performance_df, top_stock_images)
        with open(pdf_file_path, "rb") as f:
            st.download_button("Download Results as PDF", f, file_name=pdf_file_path, mime='application/pdf')

# Run the Streamlit app
if __name__ == "__main__":
    main()
