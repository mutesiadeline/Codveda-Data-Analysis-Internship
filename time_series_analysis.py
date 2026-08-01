import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
df = pd.read_csv("Stock Prices Data Set.csv")

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values('date')

# Set date as index
df.set_index('date', inplace=True)

# Select closing price
ts = df['close']

# Plot original time series
plt.figure(figsize=(12,5))
plt.plot(ts)
plt.title("Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.show()


# Time series decomposition
decomposition = seasonal_decompose(
    ts,
    model='additive',
    period=30
)

# Plot decomposition
decomposition.plot()
plt.show()


# Moving average smoothing
moving_average = ts.rolling(window=30).mean()

plt.figure(figsize=(12,5))
plt.plot(ts, label="Original")
plt.plot(moving_average, label="30-Day Moving Average")

plt.title("Moving Average Smoothing")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.show()