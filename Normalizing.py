#When normalizing detrended 1-minute OHLCVT data, it's essential to consider the different nature of each variable. Here's a suggested way to normalize each component of the OHLCVT data using Min-Max scaling and z-score standardization:

#Min-Max Scaling for price-related variables (Open, High, Low, Close):
#python
#Copy code
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


# Load your detrended OHLCVT data
data = pd.read_csv('detrended_ohlcvt_data.csv')

# Create a MinMaxScaler with the desired range for price-related variables
price_scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit the scaler to the price-related variables and transform them
price_columns = ['Open_Detrended', 'High_Detrended', 'Low_Detrended', 'Close_Detrended']
data[price_columns] = price_scaler.fit_transform(data[price_columns])



from sklearn.preprocessing import StandardScaler

# Create a StandardScaler for Volume and Trade variables
volume_trade_scaler = StandardScaler()

# Fit the scaler to the Volume and Trade variables and transform them
volume_trade_columns = ['Volume_Detrended', 'Trades_Detrended']
data[volume_trade_columns] = volume_trade_scaler.fit_transform(data[volume_trade_columns])


data.to_csv('normalized_detrended_ohlcvt_data.csv', index=False)


# Plot the normalized data
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

# Plot the price-related variables (Open_Detrended, High_Detrended, Low_Detrended, Close_Detrended)
data[price_columns].plot(ax=ax[0])
ax[0].set_title('Normalized Price-related Variables')
ax[0].set_ylabel('Normalized Value')
ax[0].legend()

# Plot the Volume and Trade variables (Volume_Detrended, Trades_Detrended)
data[volume_trade_columns].plot(ax=ax[1])
ax[1].set_title('Normalized Volume and Trade Variables')
ax[1].set_ylabel('Normalized Value')
ax[1].legend()

# Plot all variables together for comparison
data.plot(ax=ax[2])
ax[2].set_title('All Normalized Variables')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Normalized Value')
ax[2].legend()

# Display the plot
plt.tight_layout()
plt.show()
