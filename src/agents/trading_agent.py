import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("processed_data/cleaned_nifty50_all_stocks.csv")
df = df[df['Stock'].str.contains('INFY', case=False)]

prices = df['Close'].values

balance = 10000
stock_owned = 0

print("🚀 Smart Trading Agent Running...\n")

for i in range(len(prices)-1):

    current_price = prices[i]
    next_price = prices[i+1]  # using future as prediction (for now)

    # SMART DECISION
    if next_price > current_price:
        action = "BUY"
    elif next_price < current_price:
        action = "SELL"
    else:
        action = "HOLD"

    # Execute action
    if action == "BUY" and balance >= current_price:
        stock_owned += 1
        balance -= current_price

    elif action == "SELL" and stock_owned > 0:
        stock_owned -= 1
        balance += current_price

    print(f"Step {i} | {action} | Price: {current_price:.2f}")

# Final portfolio value
final_value = balance + stock_owned * prices[-1]

print("\n💰 Final Portfolio Value:", final_value)