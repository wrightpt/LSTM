import pandas as pd

# Read CSV file
filename = 'XMRUSD_1.csv'
data = pd.read_csv(filename)

# Ensure that the data is in the correct format
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades']


data['Date'] = pd.to_datetime(data['Date'])


from scipy import signal

# Detrend the OHLCVT columns
for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Trades']:
    data[column + '_Detrended'] = signal.detrend(data[column])

data.to_csv('detrended_ohlcvt_data.csv', index=False)


import matplotlib.pyplot as plt

# Plot the original and detrended data for comparison
fig, ax = plt.subplots(2, 1, figsize=(12, 6))

ax[0].plot(data['Date'], data['Close'], label='Original Close Price')
ax[0].set_title('Original Close Price')
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Price')

ax[1].plot(data['Date'], data['Close_Detrended'], label='Detrended Close Price')
ax[1].set_title('Detrended Close Price')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Detrended Price')

plt.tight_layout()
plt.show()

