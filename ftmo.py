import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM, MultiHeadAttention, LayerNormalization, Flatten, Input
from keras.callbacks import EarlyStopping
from keras.metrics import RootMeanSquaredError as rmse
import logging
import os
import time
import tensorflow as tf

ROLLOVER_HOUR = 23  
ROLLOVER_END_HOUR = 1

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
start_date = end_date - timedelta(days=1000)

# Global variables
account_balance = mt5.account_info().balance
max_daily_loss = account_balance * 0.05  # 5% daily loss limit
max_drawdown = account_balance * 0.1  # 10% overall drawdown limit
daily_loss = 0
symbols = ["USDJPY",]
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
time_step = 120
trailing_stop_enabled = True  # Enable or disable trailing stop here

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
def create_model_with_attention(time_step, conv_filters=256, conv_kernel_size=2, lstm_units=100, lstm_dropout=0.3, attention_heads=4, attention_key_dim=2, dense_units=3):
    inputs = Input(shape=(time_step, 1))
    x = Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = Dropout(lstm_dropout)(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = Dropout(lstm_dropout)(x)
    x = MultiHeadAttention(num_heads=attention_heads, key_dim=attention_key_dim)(x, x)
    x = LayerNormalization()(x)
    x = Flatten()(x)
    outputs = Dense(units=dense_units)(x)  # Output three values: predicted price, confidence, and trailing stop
    
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

# Train and save the model with early stopping
def train_and_save_model(symbol, x_train, y_train, x_test, y_test, early_stopping, time_step, conv_filters, conv_kernel_size, lstm_units, lstm_dropout, attention_heads, attention_key_dim, dense_units, batch_size, epochs):
    model = create_model_with_attention(time_step, conv_filters, conv_kernel_size, lstm_units, lstm_dropout, attention_heads, attention_key_dim, dense_units)
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size, verbose=2, callbacks=[early_stopping])
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
    trailing_stop = pred[:, 2][0]  # Predicted trailing stop
    return pred_price[0][0], confidence, trailing_stop

# Calculate TP and SL dynamically based on model prediction
def calculate_tp_sl(action_type, current_price, next_price, confidence, risk_reward_ratio=2):
    if action_type == mt5.ORDER_TYPE_BUY:
        tp_price = current_price + (next_price - current_price) * confidence * risk_reward_ratio
        sl_price = current_price - (tp_price - current_price)
    else:
        tp_price = current_price - (current_price - next_price) * confidence * risk_reward_ratio
        sl_price = current_price + (current_price - tp_price)
    return tp_price, sl_price

# Create a trading request with dynamic TP and SL
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
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if trailing_stop_enabled:
        request["trailing_stop"] = trailing_stop
    return request


def check_rollover_time():
    now = datetime.now()
    if now.hour >= ROLLOVER_HOUR and now.minute >= 0:
        logging.info("Rollover time reached. Closing all open trades.")
        
        # Retrieve all open positions
        positions = mt5.positions_get()
        
        for position in positions:
            if position.symbol in symbols:  # Check if the position belongs to the symbols list
                # Close the position
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "type": mt5.ORDER_TYPE_CLOSE,
                    "position": position.ticket,
                    "volume": position.volume,
                    "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                    "deviation": 10,
                    "magic": 234000,
                    "comment": "Closing all trades at rollover time",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                # Send the close order
                result = mt5.order_send(close_request)
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logging.error("Failed to close trade %d for symbol %s. Error code: %d", position.ticket, position.symbol, result.retcode)
                else:
                    logging.info("Closed trade %d for symbol %s", position.ticket, position.symbol)

def is_rollover_time():
    now = datetime.now()
    if now.hour >= ROLLOVER_HOUR or now.hour < ROLLOVER_END_HOUR:
        return True
    return False

# Calculate the trade volume based on 0.1% risk per trade
def calculate_volume(symbol, sl_price, current_price, risk_percentage=0.001):
    account_balance = mt5.account_info().balance
    risk_amount = account_balance * risk_percentage
    pip_value = 0.0001  # Assuming standard pip value for forex pairs
    volume = risk_amount / (abs(current_price - sl_price) * pip_value)
    
    # Ensure volume meets broker's minimum trade size or round to the nearest valid volume
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error("Failed to get symbol info for %s", symbol)
        return 0.0  # or handle the error as appropriate
    
    min_trade_volume = symbol_info.volume_min
    volume = max(volume, min_trade_volume)  # Ensure volume is at least the minimum allowed
    
    # Round volume to nearest valid lot size
    volume = round(volume / min_trade_volume) * min_trade_volume
    
    return round(volume, 2)



# Handle trading logic
def trade_logic(symbol, model, scaler):
    global daily_loss

    if is_rollover_time():
        logging.info("Rollover time. Skipping trade execution.")
        return
    
    if not check_ftmo_compliance():
        return

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, time_step + 1)
    if rates is None:
        logging.error("Failed to get current data for %s, error code = %s", symbol, mt5.last_error())
        return
    
    data = pd.DataFrame(rates).filter(['close']).values

    next_price, confidence, trailing_stop = predict_next(model, data, scaler)
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

    # Calculate TP and SL dynamically
    tp_price, sl_price = calculate_tp_sl(action_type, current_price, next_price, confidence)

    # Calculate trade volume based on risk
    volume = calculate_volume(sl_price, current_price)

    # Create trading request with dynamic TP, SL, volume, and deviation
    request = create_request(symbol, action_type, tp_price, sl_price, trailing_stop, volume=volume)

    # Send the trading request to MetaTrader5
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error("Order send failed for %s, retcode: %d", symbol, result.retcode)
        daily_loss += abs(sl_price - current_price) * volume  # Update daily loss with the risked amount
    else:
        logging.info("Order send successful for %s, order: %s", symbol, result.order)
        if trailing_stop_enabled:
            logging.info("Trailing stop enabled for %s, setting trailing stop.", symbol)
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "order": result.order,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 10,
                "magic": 234000,
                "comment": "python script set trailing stop",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
                "trailing_stop": trailing_stop, 
            })

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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = train_and_save_model(
            symbol, x_train, y_train, x_test, y_test, early_stopping, 
            time_step, 
            conv_filters=128, 
            conv_kernel_size=3, 
            lstm_units=100, 
            lstm_dropout=0.3, 
            attention_heads=4, 
            attention_key_dim=2, 
            dense_units=3,  
            batch_size=32, 
            epochs=2
        )
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

                    # Implement early stopping
                    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                    # Train and save the model
                    train_and_save_model(symbol, x_train, y_train, x_test, y_test, early_stopping)
                    models[symbol] = load_trained_model(symbol)
            
            check_rollover_time()       
            time.sleep(1)  # Sleep for a second to avoid busy waiting
    except KeyboardInterrupt:
        logging.info("Trading loop interrupted by user.")
    finally:
        mt5.shutdown()

# Kick off live trading
if __name__ == "__main__":
    trade()
