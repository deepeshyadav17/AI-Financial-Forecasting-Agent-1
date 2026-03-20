# =====================================
# 1. IMPORT LIBRARIES
# =====================================
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# =====================================
# 2. LOAD DATA
# =====================================
df = pd.read_csv("processed_data/cleaned_nifty50_all_stocks.csv")

# Clean column names (important)
df.columns = df.columns.str.strip()

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

print("✅ Data Loaded!")

# =====================================
# 3. CHECK AVAILABLE STOCKS
# =====================================
print("Available stocks:", df['Stock'].unique())

# =====================================
# 4. FILTER INFY DATA
# =====================================
df = df[df['Stock'].str.contains('INFY', case=False)]

# Sort and set index
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Take Close price
data = df['Close']

# Convert to numeric
data = pd.to_numeric(data, errors='coerce')

# Drop missing values
data = data.dropna()

print("Data after filtering:", len(data))

# =====================================
# 5. SPLIT DATA (80% train, 20% test)
# =====================================
split_index = int(len(data) * 0.8)

train = data[:split_index]
test = data[split_index:]

print("Train size:", len(train))
print("Test size:", len(test))

# =====================================
# 6. TRAIN MODEL (ARIMA)
# =====================================
model = ARIMA(train, order=(3,1,3))
model_fit = model.fit()

print("✅ Model Trained!")

# =====================================
# 7. MAKE PREDICTIONS
# =====================================
predictions = model_fit.forecast(steps=len(test))

# =====================================
# 8. CHECK ERROR (RMSE)
# =====================================
rmse = np.sqrt(mean_squared_error(test, predictions))
print("📉 RMSE:", rmse)

# =====================================
# 9. PLOT RESULTS
# =====================================
plt.figure(figsize=(10,5))

plt.plot(train, label="Train Data")
plt.plot(test, label="Real Price")
plt.plot(test.index, predictions, label="Predicted Price")

plt.legend()
plt.title("INFY Stock Prediction")
plt.xlabel("Date")
plt.ylabel("Price")

plt.grid()
plt.show()