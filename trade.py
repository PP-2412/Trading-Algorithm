import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def markov_chain_trading(stock_data, stock_name):
    """
    Analyzes stock trends using Markov chains and provides recommendations (Buy/Sell/Hold).

    Parameters:
        stock_data (pd.DataFrame): Historical stock data with 'Date' and 'Close' columns.
        stock_name (str): The stock name being analyzed.

    Returns:
        str: Recommendation - Buy, Sell, or Hold.
    """
    # Preprocessing
    stock_data['Return'] = stock_data['Close'].pct_change()  # Calculate daily returns
    stock_data.dropna(inplace=True)

    # Define states: Bull, Bear, or Neutral based on returns
    conditions = [
        (stock_data['Return'] > 0.005),
        (stock_data['Return'] < -0.005),
    ]
    choices = ['Bull', 'Bear']
    stock_data['State'] = np.select(conditions, choices, default='Neutral')

    # Encode states numerically for transition matrix
    le = LabelEncoder()
    stock_data['State_Encoded'] = le.fit_transform(stock_data['State'])

    # Calculate the transition matrix
    states = le.classes_
    n_states = len(states)
    transition_matrix = np.zeros((n_states, n_states))

    for (prev, curr) in zip(stock_data['State_Encoded'], stock_data['State_Encoded'][1:]):
        transition_matrix[prev, curr] += 1

    # Normalize the transition matrix
    with np.errstate(invalid='ignore'):
        transition_matrix = (transition_matrix.T / transition_matrix.sum(axis=1)).T
    transition_matrix = np.nan_to_num(transition_matrix)

    # Predict next state based on current state
    current_state = stock_data['State_Encoded'].iloc[-1]
    next_state_probs = transition_matrix[current_state]
    next_state = np.argmax(next_state_probs)

    # Recommendation logic
    predicted_state = le.inverse_transform([next_state])[0]
    if predicted_state == 'Bull':
        recommendation = 'Buy'
    elif predicted_state == 'Bear':
        recommendation = 'Sell'
    else:
        recommendation = 'Hold'

    return f"Stock: {stock_name} | Recommendation: {recommendation}"

if __name__ == "__main__":
    import yfinance as yf

    # Ask user for input
    stock_name = input("Enter the stock ticker (e.g., AAPL, TSLA): ").strip().upper()
    print("Fetching data...")

    try:
        # Fetch historical data for the stock
        stock = yf.Ticker(stock_name)
        stock_data = stock.history(period="1mo")[['Close']]
        stock_data.reset_index(inplace=True)

        if not stock_data.empty:
            result = markov_chain_trading(stock_data, stock_name)
            print(result)
        else:
            print(f"Error: No price data found for {stock_name}. The symbol may be delisted or incorrect.")
    except Exception as e:
        print(f"An error occurred: {e}")
