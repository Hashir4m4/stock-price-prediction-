# predictor.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data/stock_data.csv")

# Convert 'Date' to datetime and encode it as integer for ML
df['Date'] = pd.to_datetime(df['Date'])
df['DateInt'] = df['Date'].map(lambda date: date.toordinal())

# Features and Target
X = df[['DateInt']]          # Date as numerical input
y = df['Close']              # Predicting closing price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Close'], label="Actual Close")
plt.plot(df['Date'].iloc[-len(y_pred):], y_pred, label="Predicted Close", linestyle='--')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction")
plt.legend()
plt.tight_layout()
plt.show()

# Print future price prediction
future_date = df['DateInt'].max() + 1
future_pred = model.predict([[future_date]])
print("Predicted Closing Price for Next Day:", round(future_pred[0], 2))
