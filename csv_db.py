import os
import pandas as pd
from datetime import datetime

# Folder to store CSV files
CSV_FOLDER = 'csv_cache'

def initialize_db():
    """
    Checks if the CSV folder exists. If not, creates a new folder.
    """
    if not os.path.exists(CSV_FOLDER):
        os.makedirs(CSV_FOLDER)
        print(f"Created CSV folder: {CSV_FOLDER}")

def summary_file_path(ticker):
    """
    Constructs the file path for the summary CSV for a given ticker.
    """
    return os.path.join(CSV_FOLDER, f"{ticker}.csv")

def file_exists(ticker):
    """
    Returns True if the summary CSV file for the given ticker exists.
    """
    return os.path.exists(summary_file_path(ticker))

def read_stock_cache(ticker):
    if file_exists(ticker):
        df = pd.read_csv(summary_file_path(ticker))
        if not df.empty:
            result = df.iloc[-1].to_dict()
            # Remove keys that weren't meant to be stored in CSV.
            result.pop('Train Data', None)
            result.pop('Test Data', None)
            return result
    return None


def update_stock_cache(ticker, analysis_result):
    """
    Updates the summary cache for a ticker.
    If the CSV file exists and the latest row is not from today, it appends a new row.
    Otherwise, it updates the last row.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    analysis_result['Date'] = today
    path = summary_file_path(ticker)
    
    if file_exists(ticker):
        df = pd.read_csv(path)
        if not df.empty and df.iloc[-1].get('Date') == today:
            # Update the row if the analysis was already done today.
            df.iloc[-1] = pd.Series(analysis_result)
        else:
            # Append a new row using pd.concat instead of .append
            new_row = pd.DataFrame([analysis_result])
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        # Create a new DataFrame for this ticker.
        df = pd.DataFrame([analysis_result])
    
    # Write the updated DataFrame to the CSV file.
    df.to_csv(path, index=False)


#############################################
# Functions for Technical Data Cache
#############################################

def train_file_path(ticker):
    """
    Constructs the file path for the ticker's training data CSV.
    """
    return os.path.join(CSV_FOLDER, f"{ticker}_train.csv")

def test_file_path(ticker):
    """
    Constructs the file path for the ticker's test data CSV.
    """
    return os.path.join(CSV_FOLDER, f"{ticker}_test.csv")

def update_technical_cache(ticker, train_data, test_data):
    """
    Writes the technical indicator data (train and test DataFrames) to CSV files.
    """
    train_data.to_csv(train_file_path(ticker), index=False)
    test_data.to_csv(test_file_path(ticker), index=False)

def read_technical_cache(ticker):
    """
    Reads the cached technical data (training and test DataFrames) for the ticker.
    Returns a dictionary with keys 'Train Data' and 'Test Data' if both files exist, else None.
    """
    train_path = train_file_path(ticker)
    test_path = test_file_path(ticker)
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return {'Train Data': train_data, 'Test Data': test_data}
    return None
