import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM, MultiHeadAttention, LayerNormalization
from keras.metrics import RootMeanSquaredError as rmse
from sklearn.cluster import DBSCAN
import logging
import os
import tensorflow.keras.backend as K

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MetaTrader5 for history data
if not mt5.initialize():
    logging.error("initialize() failed, error code = %s", mt5.last_error())
    quit()

terminal_info = mt5.terminal_info()

# Show file path
file_path = terminal_info.data_path + "\\MQL5\\Files\\"
model_path = os.path.join('AUDUSD_model.keras')

# Set start and end dates for history data
end_date = datetime.now()
start_date = end_date - timedelta(days=120)

# Get AUDUSD rates (15M) from start_date to end_date
AUDUSD_rates = mt5.copy_rates_range("AUDUSD", mt5.TIMEFRAME_M15, start_date, end_date)

# Create dataframe
df = pd.DataFrame(AUDUSD_rates)

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

# Define the model with multi-head attention mechanism
def create_model_with_attention():
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same', input_shape=(120, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(MultiHeadAttention(num_heads=4, key_dim=2))
    model.add(LayerNormalization())
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse', metrics=[rmse()])
    return model

# Train and save the model
def train_and_save_model():
    model = create_model_with_attention()
    history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), batch_size=32, verbose=2)
    model.save(model_path)
    logging.info("Model trained and saved.")
    return history

# Load the model
def load_trained_model():
    return load_model(model_path, custom_objects={'MultiHeadAttention': MultiHeadAttention})

# Function to predict next value
def predict_next(model, data):
    scaled_data = scaler.transform(data)  # Scale the input data
    x_input = scaled_data[-time_step:].reshape(1, time_step, 1)  # Reshape for model input
    pred = model.predict(x_input)  # Get the model's prediction (scaled value)
    pred_price = scaler.inverse_transform(pred)  # Inverse transform to get the actual price
    return pred_price[0][0]  # Return the actual predicted price

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
def trade_logic(model):
    rates = mt5.copy_rates_from_pos("AUDUSD", mt5.TIMEFRAME_M15, 0, time_step + 1)
    data = pd.DataFrame(rates).filter(['close']).values

    next_price = predict_next(model, data)
    current_price = data[-1][0]

    latest_scaled_data = scaler.transform(data)
    dbscan = DBSCAN(eps=0.1, min_samples=10)
    latest_labels = dbscan.fit_predict(latest_scaled_data)

    latest_label = latest_labels[-1]

    if latest_label == -1:
        logging.info("Current data point is noise according to DBSCAN, skipping trade.")
        return
        # Determine the trade action based on predicted price
    if next_price > current_price:
        logging.info("Predicted price is higher than current price, placing a buy order.")
        request = create_request("AUDUSD", mt5.ORDER_TYPE_BUY)
    else:
        logging.info("Predicted price is lower than current price, placing a sell order.")
        request = create_request("AUDUSD", mt5.ORDER_TYPE_SELL)

    # Send the trading request to MetaTrader5
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error("Order send failed, retcode: %d", result.retcode)
    else:
        logging.info("Order send successful, order: %s", result.order)

# Live trading loop
def trade():
    try:
        model = load_trained_model()
        while True:
            now = datetime.now()
            # Execute trading logic every 15 minutes
            if now.minute % 15 == 0 and now.second == 0:
                logging.info("Executing trading logic.")
                trade_logic(model)
            # Retrain the model daily at midnight
            if now.hour == 0 and now.minute == 0 and now.second == 0:
                logging.info("Retraining the model at midnight.")
                train_and_save_model()
                model = load_trained_model()
            time.sleep(1)  # Sleep for a second to avoid busy waiting
    except KeyboardInterrupt:
        logging.info("Trading loop interrupted by user.")
    finally:
        mt5.shutdown()

# Kick off live trading
if __name__ == "__main__":
    train_and_save_model()  # Initial training
    trade()

   
