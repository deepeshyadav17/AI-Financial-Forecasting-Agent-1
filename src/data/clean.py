import pandas as pd
import os

# Create output folder
os.makedirs("processed_data", exist_ok=True)

DATA_FOLDER = "data"
OUTPUT_FILE = "processed_data/cleaned_nifty50_all_stocks.csv"

all_data = []

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".csv"):
        file_path = os.path.join(DATA_FOLDER, file)
        stock_name = file.replace(".csv", "")

        print(f"\nProcessing: {stock_name}")

        df = pd.read_csv(file_path)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Check Date column
        if 'Date' not in df.columns:
            print("❌ Skipping (no Date column)")
            continue

        # Convert Date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        # Sort
        df = df.sort_values('Date')

        # Convert numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill missing values
        df = df.ffill().dropna()

        # ✅ Add stock name
        df['Stock'] = stock_name

        print("Rows:", len(df))  # debug

        all_data.append(df)

# Combine all
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Cleaning completed!")
    print("Total rows:", len(final_df))
else:
    print("\n❌ No data processed!")