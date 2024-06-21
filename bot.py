# Python libraries
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
import tf2onnx
from sklearn.model_selection import train_test_split
from sys import argv
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.metrics import RootMeanSquaredError as rmse
from keras import optimizers
from sklearn.cluster import DBSCAN
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MetaTrader5 for history data
if not mt5.initialize():
    logging.error("initialize() failed, error code = %s", mt5.last_error())
    quit()

terminal_info = mt5.terminal_info()

# Show file path
file_path = terminal_info.data_path + "\\MQL5\\Files\\"

# Set start and end dates for history data
end_date = datetime.now()
start_date = end_date - timedelta(days=120)

# Get EURUSD rates (15M) from start_date to end_date
eurusd_rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M15, start_date, end_date)

# Create dataframe
df = pd.DataFrame(eurusd_rates)

# Prepare close prices only
data = df.filter(['close']).values

# Scale data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Training size is 80% of the data
training_size = int(len(scaled_data) * 0.80)

# Create train data
train_data_initial = scaled_data[0:training_size, :]

# Create test data
test_data_initial = scaled_data[training_size:, :1]

# Split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Split into samples
time_step = 120
x_train, y_train = split_sequence(train_data_initial, time_step)
x_test, y_test = split_sequence(test_data_initial, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Define the model
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same', input_shape=(120, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=[rmse()])

# Train the model
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), batch_size=32, verbose=2)

# Save the model
model.save(file_path + 'eurusd_model.keras')

# Function to predict next value
def predict_next(data):
    scaled_data = scaler.transform(data)
    x_input = scaled_data[-time_step:].reshape(1, time_step, 1)
    pred = model.predict(x_input)
    pred_price = scaler.inverse_transform(pred)
    return pred_price[0][0]

# Function to create a trading request
def create_request(symbol, action_type, volume=0.1, deviation=10, sl_pips=50, tp_pips=100):
    price = mt5.symbol_info_tick(symbol).ask if action_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    sl = price - sl_pips * mt5.symbol_info(symbol).point if action_type == mt5.ORDER_TYPE_BUY else price + sl_pips * mt5.symbol_info(symbol).point
    tp = price + tp_pips * mt5.symbol_info(symbol).point if action_type == mt5.ORDER_TYPE_BUY else price - tp_pips * mt5.symbol_info(symbol).point
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": action_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return request

# Function to handle trading logic
def trade_logic():
    # Get latest data
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, time_step + 1)
    data = pd.DataFrame(rates).filter(['close']).values

    # Predict next price
    next_price = predict_next(data)

    # Get current price
    current_price = data[-1][0]

    # Scale the latest data
    latest_scaled_data = scaler.transform(data)

    # Apply DBSCAN to the latest scaled data
    dbscan = DBSCAN(eps=0.1, min_samples=10)
    latest_label = dbscan.fit_predict(latest_scaled_data.reshape(-1, 1))[-1]

    # Simple strategy example
    if latest_label == -1:
        # If the latest point is considered noise, do not trade
        logging.info("Current data point is noise according to DBSCAN, skipping trade.")
        return
    
    if next_price > current_price:
        # Buy signal
        request = create_request("EURUSD", mt5.ORDER_TYPE_BUY)
    else:
        # Sell signal
        request = create_request("EURUSD", mt5.ORDER_TYPE_SELL)

    # Send trading request
    result = mt5.order_send(request)
    logging.info(result)

# Live trading loop
def trade():
    try:
        while True:
            now = datetime.now()
            if now.minute % 15 == 0 and now.second == 0:  # Check every 15 minutes
                trade_logic()
            time.sleep(1)  # Sleep for a second to avoid busy waiting
    except KeyboardInterrupt:
        logging.info("Trading loop interrupted by user.")
    finally:
        mt5.shutdown()

# Kick off live trading
if __name__ == "__main__":
    trade()
