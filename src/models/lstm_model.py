# =====================================
# 1. IMPORT LIBRARIES
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# =====================================
# 2. LOAD DATA
# =====================================
df = pd.read_csv("processed_data/cleaned_nifty50_all_stocks.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Stock'].str.contains('INFY', case=False)]

df = df.sort_values('Date')
data = df['Close'].values.reshape(-1,1)

print("✅ Data Loaded")

# =====================================
# 3. SCALE DATA (IMPORTANT)
# =====================================
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

# =====================================
# 4. CREATE SEQUENCES
# =====================================
X = []
y = []

window_size = 60

for i in range(window_size, len(data_scaled)):
    X.append(data_scaled[i-window_size:i])
    y.append(data_scaled[i])

X, y = np.array(X), np.array(y)

# =====================================
# 5. TRAIN-TEST SPLIT
# =====================================
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================================
# 6. BUILD LSTM MODEL
# =====================================
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

print("✅ Training model...")
model.fit(X_train, y_train, epochs=5, batch_size=32)

# =====================================
# 7. PREDICTION
# =====================================
predictions = model.predict(X_test)

# Convert back to original scale
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# =====================================
# 8. PLOT
# =====================================
plt.figure(figsize=(10,5))

plt.plot(y_test, label="Real Price")
plt.plot(predictions, label="Predicted Price")

plt.legend()
plt.title("LSTM Stock Prediction")
plt.show()
# Save predictions
results = pd.DataFrame({
    "Real": y_test.flatten(),
    "Predicted": predictions.flatten()
})

results.to_csv("processed_data/lstm_predictions.csv", index=False)

print("✅ Predictions saved!")