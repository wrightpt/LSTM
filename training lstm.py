import pandas as pd

import matplotlib.pyplot as plt

import joblib




from sklearn.model_selection import train_test_split
data = pd.read_csv('normalized_detrended_ohlcvt_data.csv')



# Define the price_scaler and price_columns variables according to your data
# For example:
from sklearn.preprocessing import MinMaxScaler
price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
price_scaler = MinMaxScaler()
price_scaler.fit(data[price_columns])


# Define the target variable (e.g., 'Close_Detrended')
target = 'Close_Detrended'

X_train, X_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target], test_size=0.2, shuffle=False)


import numpy as np

window_size = 60

# def create_sequences(X, y, window_size):
#     X_seq, y_seq = [], []
#     for i in range(len(X) - window_size):
#         X_seq.append(X[i:i + window_size])
#         y_seq.append(y[i + window_size])
#     return np.array(X_seq), np.array(y_seq)

def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# Drop non-numeric columns, such as 'Date' or 'Timestamp'
numeric_columns = data.select_dtypes(include=[np.number]).columns
data = data[numeric_columns]

# Then proceed with the train-test split
X_train, X_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target], test_size=0.2, shuffle=False)


X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, window_size)
X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, window_size)

from tensorflow import keras


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, X_train.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile and train the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
# history = model.fit(X_train_seq, y_train_seq, epochs=2, batch_size=64, validation_split=0.1, verbose=1)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Add callbacks to the model.fit() function
history = model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=64, validation_split=0.1, verbose=1, callbacks=[early_stopping, model_checkpoint])


# # Evaluate the model
# loss, mae = model.evaluate(X_test_seq, y_test_seq)
# print("Mean Absolute Error:", mae)
#
# # Make predictions
# predictions = model.predict(X_test_seq)

from tensorflow.keras.models import load_model

# Load the best model
best_model = load_model('best_model.h5')

# Evaluate the best model
loss, mae = best_model.evaluate(X_test_seq, y_test_seq)
print("Mean Absolute Error:", mae)

# Make predictions using the best model
predictions = best_model.predict(X_test_seq)



# Save the scaler
joblib.dump(price_scaler, 'price_scaler.pkl')

# Load the scaler
price_scaler = joblib.load('price_scaler.pkl')






# Inverse transform the predictions
predicted_prices = price_scaler.inverse_transform(np.concatenate((X_test.iloc[window_size:][price_columns[:-1]], predictions), axis=1))[:, -1]

# Create a DataFrame with actual and predicted prices
results = pd.DataFrame({'Actual': y_test.iloc[window_size:].values, 'Predicted': predicted_prices})


from tensorflow.keras.models import load_model

# Load the best model
best_model = load_model('best_model.h5')

# Evaluate the best model
loss, mae = best_model.evaluate(X_test_seq, y_test_seq)
print("Mean Absolute Error:", mae)

# Make predictions using the best model
predictions = best_model.predict(X_test_seq)


# Plot actual vs. predicted prices
results.plot(figsize=(12, 6), title='Actual vs. Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
