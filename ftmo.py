import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM, MultiHeadAttention, LayerNormalization
from keras.callbacks import EarlyStopping, Callback
from keras.metrics import RootMeanSquaredError as rmse
import logging
import os
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dropout, MultiHeadAttention, Dense, LayerNormalization, Flatten
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
symbols = ["EURUSD","USDJPY","GPBUSD","AUDUSD" ]
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

# Define the model with multi-head attention mechanism
def create_model_with_attention(time_step):
    inputs = Input(shape=(time_step, 1))
    x = Conv1D(filters=256, kernel_size=2, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(100, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(100, return_sequences=True)(x)
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
    model = create_model_with_attention(time_step)
    
    history = model.fit(
        x_train, y_train,
        epochs=50,
        validation_data=(x_test, y_test),
        batch_size=32,
        verbose=0,
        callbacks=[early_stopping, TqdmCallback(verbose=1), MetricsCallback()]
    )
    
    model.save(os.path.join(model_dir, f"{symbol}_model.keras"))
    logging.info("Model for %s trained and saved.", symbol)
    return history

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

# Calculate TP and SL dynamically based on model prediction
def calculate_tp_sl(action_type, current_price, next_price, confidence, risk_reward_ratio=2):
    if action_type == mt5.ORDER_TYPE_BUY:
        tp_price = current_price + (next_price - current_price) * confidence * risk_reward_ratio
        sl_price = current_price - (tp_price - current_price)
    else:
        tp_price = current_price - (current_price - next_price) * confidence * risk_reward_ratio
        sl_price = current_price + (current_price - tp_price)
    return tp_price, sl_price

# Calculate trailing stop-loss dynamically based on confidence
def calculate_trailing_stop_loss(action_type, current_price, confidence, min_tsl=10, max_tsl=50):
    tsl_range = max_tsl - min_tsl
    tsl = min_tsl + tsl_range * confidence
    if action_type == mt5.ORDER_TYPE_BUY:
        tsl_price = current_price - tsl
    else:
        tsl_price = current_price + tsl
    return tsl_price

# Create a trading request with dynamic TP and SL
def create_request(symbol, action_type, tp_price, sl_price, deviation=10, volume=0.01):
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
        "comment": "lstm v1",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return request

# Function to close all open trades
def close_all_trades():
    orders = mt5.positions_get()
    if orders is None:
        logging.error("Failed to get open positions, error code = %s", mt5.last_error())
        return
    
    for order in orders:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": mt5.ORDER_TYPE_SELL if order.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": order.ticket,
            "price": mt5.symbol_info_tick(order.symbol).bid if order.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(order.symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close all trades",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error("Failed to close order %d, retcode: %d", order.ticket, result.retcode)
        else:
            logging.info("Order %d closed successfully.", order.ticket)


# Handle trading logic
def trade_logic(symbol, model, scaler):
    global daily_loss
    if not check_ftmo_compliance():
        return

    now = datetime.now()
    if 23 <= now.hour or now.hour < 1:
        logging.info("Rollover period detected, skipping trade.")
        close_all_trades()
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
        logging.info("Low confidence in price movement for %s, skipping trade.", symbol)
        return

    tp_price, sl_price = calculate_tp_sl(action_type, current_price, next_price, confidence)
    tsl_price = calculate_trailing_stop_loss(action_type, current_price, confidence)

    risk_per_trade = account_balance * 0.01  # Risk 1% of account balance per trade
    volume = risk_per_trade / abs(sl_price - current_price)

    if confidence > 0.7:
        volume = risk_per_trade / abs(sl_price - current_price)  # Higher confidence, higher volume
    elif confidence > 0.5:
        volume = (risk_per_trade / abs(sl_price - current_price)) / 2  # Moderate confidence, standard volume
    else:
        volume = (risk_per_trade / abs(sl_price - current_price)) / 4  # Low confidence, lower volume

    deviation = 20 

    request = create_request(symbol, action_type, tp_price, tsl_price, deviation=deviation, volume=volume)

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error("Order send failed for %s, retcode: %d", symbol, result.retcode)
        daily_loss += abs(sl_price - current_price) * volume
    else:
        logging.info("Order send successful for %s, order: %s", symbol, result.order)

# Live trading loop
def trade():
    models = {}
    scalers = {}

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

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = train_and_save_model(symbol, x_train, y_train, x_test, y_test, early_stopping, time_step)
        models[symbol] = load_trained_model(symbol)

    try:
        while True:
            now = datetime.now()
            if now.minute % 15 == 0 and now.second == 0:
                for symbol in symbols:
                    if symbol in models and models[symbol] is not None:
                        logging.info("Executing trading logic for %s.", symbol)
                        trade_logic(symbol, models[symbol], scalers[symbol])
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

                    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                    train_and_save_model(symbol, x_train, y_train, x_test, y_test, early_stopping)
                    models[symbol] = load_trained_model(symbol)
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Trading loop interrupted by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    trade()
