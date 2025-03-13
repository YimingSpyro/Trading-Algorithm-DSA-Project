"""
visualization_and_report.py

This module provides functions for visualizing stock data with technical indicators and 
generating reports. It includes functionalities to plot stock charts displaying various 
technical indicators such as Bollinger Bands, MACD, RSI, and Buy/Sell signals, and to 
save the plots as images if needed. The module is designed to support both on-screen 
visualization using Streamlit and offline report generation.

Functions:
    - plot_or_save_stock_data: Plots stock data with indicators, and optionally saves the 
      plot as an image file.
"""

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF 
import os
import streamlit as st

def plot_or_save_stock_data(ticker, test_data, action='show'):
    # Ensure the DataFrame is a copy to avoid view/copy issues
    test_data = test_data.copy()

    # Check if the necessary columns exist
    required_columns = ['Date', 'Close', 'Upper_Band', 'Lower_Band', 'MACD', 'Signal', 'RSI', 'Buy_Signal', 'Sell_Signal']
    for column in required_columns:
        if column not in test_data.columns:
            raise ValueError(f"The '{column}' column is missing from the test data.")

    # Convert 'Date' to datetime if not already
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    
    # Set 'Date' as the index
    test_data.set_index('Date', inplace=True)

    # Prepare data for plotting
    upper_band = test_data['Upper_Band']
    lower_band = test_data['Lower_Band']
    sma_20 = test_data['Close'].rolling(window=20).mean()
    macd = test_data['MACD']
    signal = test_data['Signal']
    rsi = test_data['RSI']
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(14, 10), sharex=True)

    # Plot closing price, Bollinger Bands, Moving Averages, and Buy/Sell signals
    ax1.plot(test_data['Close'], label='Close Price', color='black')
    ax1.plot(upper_band, label='Upper Bollinger Band', color='red')
    ax1.plot(lower_band, label='Lower Bollinger Band', color='blue')
    ax1.plot(sma_20, label='20-day SMA', color='orange')
    ax1.fill_between(test_data.index, lower_band, upper_band, color='gray', alpha=0.3)
    ax1.scatter(test_data.index[test_data['Buy_Signal']], 
                test_data['Close'][test_data['Buy_Signal']], 
                label='Buy Signal', marker='^', color='green', alpha=1)
    ax1.scatter(test_data.index[test_data['Sell_Signal']], 
                test_data['Close'][test_data['Sell_Signal']], 
                label='Sell Signal', marker='v', color='red', alpha=1)
    ax1.set_title(f'{ticker} Stock Price, Bollinger Bands, Moving Averages, Buy & Sell Signals')
    ax1.legend(loc='upper left')  # Move legend to the top left

    # Plot MACD and Signal Line
    ax2.plot(macd, label='MACD', color='green')
    ax2.plot(signal, label='Signal Line', color='red')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('MACD and Signal Line')
    ax2.legend(loc='upper left')  # Move legend to the top left

    # Plot RSI
    ax3.plot(rsi, label='RSI', color='purple')
    ax3.axhline(30, linestyle='--', alpha=0.5, color='red')
    ax3.axhline(70, linestyle='--', alpha=0.5, color='red')
    ax3.set_title('RSI')
    ax3.legend(loc='upper left')  # Move legend to the top left

    # Use Streamlit to display plots
    if action == 'save':
        # Ensure the output directory exists
        output_dir = 'saved_plots'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_path = os.path.join(output_dir, f'{ticker}_signals.png')
        plt.savefig(image_path)
        plt.close()
        return image_path
    else:
        # Display the plot in Streamlit
        st.pyplot(fig)
        plt.close()  # Close the figure after displaying to avoid overlapping plots

# Function to download results as PDF
def download_results_as_pdf(stock_ranking_df, performance_df, top_stock_images):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Stock Analysis and Trading Strategy Results', ln=True, align='C')
    pdf.ln(10)

    # Stock Ranking Table
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Stock Ranking:', ln=True)
    pdf.set_font("Arial", '', 12)

    # Calculate column widths for stock ranking table
    total_width = pdf.w - 2 * pdf.l_margin  # Total width available for content
    col_widths_ranking = [30, 50, 40, 40, 50]  # Widths for each column in stock ranking
    col_widths_ranking = [total_width * (width / sum(col_widths_ranking)) for width in col_widths_ranking]

    # Table header
    headers = ['Rank', 'Ticker', 'Current Price ($)', 'Target Price ($)', 'Potential Upside (%)']
    for i, header in enumerate(headers):
        pdf.cell(col_widths_ranking[i], 10, header, 1)
    pdf.ln()

    # Table rows
    for index, row in stock_ranking_df.iterrows():
        pdf.cell(col_widths_ranking[0], 10, str(int(row['Rank'])), 1)
        pdf.cell(col_widths_ranking[1], 10, row['Ticker'], 1)
        pdf.cell(col_widths_ranking[2], 10, str(round(row['Current Price ($)'], 2)), 1)
        pdf.cell(col_widths_ranking[3], 10, str(round(row['Target Price ($)'], 2)), 1)
        pdf.cell(col_widths_ranking[4], 10, str(round(row['Potential Upside (%)'], 2)), 1)
        pdf.ln()

    pdf.ln(10)

    # Trade Bot Performance Table
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Trade Bot Performance:', ln=True)
    pdf.set_font("Arial", '', 12)

    # Calculate column widths for trade bot performance table
    col_widths_performance = [30, 50, 50, 60, 50]  # Widths for each column in performance table
    col_widths_performance = [total_width * (width / sum(col_widths_performance)) for width in col_widths_performance]

    # Table header
    headers_performance = ['Rank', 'Ticker', 'Trades Closed', 'Total Return (%)', 'Status']
    for i, header in enumerate(headers_performance):
        pdf.cell(col_widths_performance[i], 10, header, 1)
    pdf.ln()

    # Table rows
    for index, row in performance_df.iterrows():
        pdf.cell(col_widths_performance[0], 10, str(int(row['Rank'])), 1)
        pdf.cell(col_widths_performance[1], 10, row['Ticker'], 1)
        pdf.cell(col_widths_performance[2], 10, str(row['Trades Closed']), 1)
        pdf.cell(col_widths_performance[3], 10, str(round(row['Total Return (%)'], 2)), 1)
        pdf.cell(col_widths_performance[4], 10, str(row['Status']), 1)
        pdf.ln()

    pdf.ln(10)

    # Visualisation analysis of Stocks Section
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Stock Analysis: Ranked by highest returns', ln=True)
    pdf.set_font("Arial", '', 12)

    for image_path in top_stock_images:
        pdf.image(image_path, x=10, w=pdf.w - 20)  # Adjust width to fit the page width, leave margin
        pdf.ln(5)  # Add space between images

    returns_list = stock_ranking_df['Ticker'].tolist()
    stock_name = "_".join(returns_list)
    pdf_file_path = f"{stock_name}_stock_analysis_results.pdf"

    pdf.output(pdf_file_path)

    # Remove saved stock data images
    for image_path in top_stock_images:
        os.remove(image_path)

    return pdf_file_path