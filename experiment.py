import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from keras.models import load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM, Bidirectional, Flatten, MultiHeadAttention, LayerNormalization
from keras.callbacks import EarlyStopping, Callback
from keras.metrics import RootMeanSquaredError as rmse
import logging
import os
import time
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tqdm.keras import TqdmCallback
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MetaTrader5
def initialize_mt5():
    if not mt5.initialize():
        logging.error("initialize() failed, error code = %s", mt5.last_error())
        quit()

initialize_mt5()

# Set start and end dates for history data
end_date = datetime.now()
start_date = end_date - timedelta(days=120)

# Global variables
account_balance = mt5.account_info().balance
max_daily_loss = account_balance * 0.05  # 5% daily loss limit
max_drawdown = account_balance * 0.1  # 10% overall drawdown limit
daily_loss = 0
symbols = ["USDJPY"]
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
time_step = 120

# Function to check FTMO rules compliance
def check_ftmo_compliance():
    global daily_loss
    equity = mt5.account_info().equity
    balance = mt5.account_info().balance

    if daily_loss >= max_daily_loss:
        logging.warning("Daily loss limit reached. Stopping trading for the day.")
        return False
    if balance - equity >= max_drawdown:
        logging.warning("Maximum drawdown limit reached. Stopping trading.")
        return False
    return True

# Get historical data
def get_historical_data(symbol):
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start_date, end_date)
    if rates is None:
        logging.error("Failed to get historical data for %s, error code = %s", symbol, mt5.last_error())
        return None
    return pd.DataFrame(rates)

# Prepare data
def prepare_data(df):
    data = df.filter(['close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        X.append(sequence[i:end_ix])
        y.append(sequence[end_ix])
    return np.array(X), np.array(y)

# Define the model with stacked bidirectional LSTM layers
def create_model_with_bilstm(time_step):
    inputs = Input(shape=(time_step, 1))
    x = Conv1D(filters=256, kernel_size=2, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(100, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(100, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = MultiHeadAttention(num_heads=4, key_dim=2)(x, x)
    x = LayerNormalization()(x)
    x = Flatten()(x)
    outputs = Dense(units=2)(x)  # Output two values: predicted price and confidence
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        tqdm.write(f"Epoch {epoch + 1} - loss: {logs.get('loss'):.4f} - val_loss: {logs.get('val_loss'):.4f} - rmse: {logs.get('root_mean_squared_error'):.4f} - val_rmse: {logs.get('val_root_mean_squared_error'):.4f}")

# Train and save the model with early stopping
def train_and_save_model(symbol, x_train, y_train, x_test, y_test, early_stopping, time_step):
    model = create_model_with_bilstm(time_step)
    
    history = model.fit(
        x_train, y_train,
        epochs=50,
        validation_data=(x_test, y_test),
        batch_size=32,
        verbose=0,
        callbacks=[early_stopping, TqdmCallback(verbose=1), MetricsCallback()]
    )
    
    model_path = os.path.join(model_dir, f"{symbol}_model.keras")
    model.save(model_path)
    logging.info("Model saved for %s at %s", symbol, model_path)

# Load the model
def load_trained_model(symbol):
    model_path = os.path.join(model_dir, f"{symbol}_model.keras")
    try:
        model = load_model(model_path, custom_objects={'MultiHeadAttention': MultiHeadAttention})
        return model
    except Exception as e:
        logging.error("Failed to load model for %s: %s", symbol, e)
        return None

# Predict next value
def predict_next(model, data, scaler):
    scaled_data = scaler.transform(data)  # Scale the input data
    x_input = scaled_data[-time_step:].reshape(1, time_step, 1)  # Reshape for model input
    pred = model.predict(x_input)  # Get the model's prediction (scaled value)
    pred_price = scaler.inverse_transform(pred[:, 0].reshape(-1, 1))  # Inverse transform to get the actual price
    confidence = pred[:, 1][0]  # Confidence in prediction
    return pred_price[0][0], confidence

# Calculate TP, SL, and Trailing Stop dynamically based on model prediction
def calculate_tp_sl_trailing(action_type, current_price, next_price, confidence, risk_reward_ratio=2):
    if action_type == mt5.ORDER_TYPE_BUY:
        tp_price = current_price + (next_price - current_price) * confidence * risk_reward_ratio
        sl_price = current_price - (tp_price - current_price)
        trailing_stop = current_price * 0.001  # Example: 0.1% trailing stop
    else:
        tp_price = current_price - (current_price - next_price) * confidence * risk_reward_ratio
        sl_price = current_price + (current_price - tp_price)
        trailing_stop = current_price * 0.001  # Example: 0.1% trailing stop
    return tp_price, sl_price, trailing_stop

# Create a trading request with dynamic TP, SL, and Trailing Stop
def create_request(symbol, action_type, tp_price, sl_price, trailing_stop, deviation=10, volume=0.01):
    if action_type == mt5.ORDER_TYPE_BUY:
        price = mt5.symbol_info_tick(symbol).ask
    else:
        price = mt5.symbol_info_tick(symbol).bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": action_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": deviation,
        "magic": 234000,
        "comment": "LSTM",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "trailing_stop": trailing_stop,
    }
    return request

# Handle trading logic
def trade_logic(symbol, model, scaler):
    global daily_loss
    if not check_ftmo_compliance():
        return

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, time_step + 1)
    if rates is None:
        logging.error("Failed to get current data for %s, error code = %s", symbol, mt5.last_error())
        return
    
    data = pd.DataFrame(rates).filter(['close']).values

    next_price, confidence = predict_next(model, data, scaler)
    current_price = data[-1][0]

    latest_scaled_data = scaler.transform(data)
    dbscan = DBSCAN(eps=0.1, min_samples=10)
    latest_labels = dbscan.fit_predict(latest_scaled_data)

    latest_label = latest_labels[-1]

    if latest_label == -1:
        logging.info("Current data point for %s is noise according to DBSCAN, skipping trade.", symbol)
        return

    # Determine the trade action based on predicted price and confidence
    if next_price > current_price and confidence > 0.5:
        logging.info("Predicted price for %s is higher than current price with confidence, placing a buy order.", symbol)
        action_type = mt5.ORDER_TYPE_BUY
    elif next_price < current_price and confidence > 0.5:
        logging.info("Predicted price for %s is lower than current price with confidence, placing a sell order.", symbol)
        action_type = mt5.ORDER_TYPE_SELL
    else:
        logging.info("Not confident in the prediction for %s, skipping trade.", symbol)
        return

    # Calculate TP, SL, and Trailing Stop dynamically
    tp_price, sl_price, trailing_stop = calculate_tp_sl_trailing(action_type, current_price, next_price, confidence)

    # Set volume based on confidence (example values, adjust as needed)
    if confidence > 0.7:
        volume = 0.02  # High confidence, higher volume
    elif confidence > 0.5:
        volume = 0.01  # Moderate confidence, standard volume
    else:
        volume = 0.005  # Low confidence, lower volume

    # Set deviation based on price movement (example, adjust as needed)
    predicted_price = next_price if action_type == mt5.ORDER_TYPE_BUY else current_price
    expected_price_movement = abs(predicted_price - current_price)
    deviation = int(expected_price_movement / 2)  # Adjust based on model's output

    # Create trading request with dynamic TP, SL, volume, deviation, and trailing stop
    request = create_request(symbol, action_type, tp_price, sl_price, trailing_stop, deviation=deviation, volume=volume)

    # Send the trading request to MetaTrader5
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error("Order send failed for %s, retcode: %d", symbol, result.retcode)
        daily_loss += abs(sl_price - current_price) * volume  # Update daily loss with the risked amount
    else:
        logging.info("Order send successful for %s, order: %s", symbol, result.order)

# Live trading loop
def trade():
    models = {}
    scalers = {}

    # Train and load models for each symbol
    for symbol in symbols:
        df = get_historical_data(symbol)
        if df is None:
            continue
        scaled_data, scaler = prepare_data(df)
        scalers[symbol] = scaler

        training_size = int(len(scaled_data) * 0.80)
        train_data_initial = scaled_data[:training_size]
        test_data_initial = scaled_data[training_size:]

        x_train, y_train = split_sequence(train_data_initial, time_step)
        x_test, y_test = split_sequence(test_data_initial, time_step)

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # Implement early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train and save the model
        train_and_save_model(symbol, x_train, y_train, x_test, y_test, early_stopping, time_step)
        models[symbol] = load_trained_model(symbol)

    try:
        while True:
            now = datetime.now()
            # Execute trading logic every 15 minutes for each symbol
            if now.minute % 15 == 0 and now.second == 0:
                for symbol in symbols:
                    if symbol in models and models[symbol] is not None:
                        logging.info("Executing trading logic for %s.", symbol)
                        trade_logic(symbol, models[symbol], scalers[symbol])
            # Retrain the model daily at midnight
            if now.hour == 0 and now.minute == 0 and now.second == 0:
                for symbol in symbols:
                    df = get_historical_data(symbol)
                    if df is None:
                        continue
                    scaled_data, scaler = prepare_data(df)
                    scalers[symbol] = scaler

                    training_size = int(len(scaled_data) * 0.80)
                    train_data_initial = scaled_data[:training_size]
                    test_data_initial = scaled_data[training_size:]

                    x_train, y_train = split_sequence(train_data_initial, time_step)
                    x_test, y_test = split_sequence(test_data_initial, time_step)

                    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

                    # Train and save the model
                    train_and_save_model(symbol, x_train, y_train, x_test, y_test, early_stopping, time_step)
                    models[symbol] = load_trained_model(symbol)

            # Sleep for a while to avoid tight loop
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Trading stopped manually.")
    finally:
        mt5.shutdown()

# Start trading
if __name__ == "__main__":
    trade()
