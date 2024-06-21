import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
import joblib
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, LSTM, Layer
from keras.metrics import RootMeanSquaredError as rmse
from sklearn.cluster import DBSCAN
import logging
import os
# import keras.backend as K
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
model_path = os.path.join('eurusd_model.keras')
svm_model_path = os.path.join('svm_model.pkl')
scaler_path = os.path.join('scaler.pkl')

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

# Define the Attention layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

# Define the model with attention mechanism
def create_model_with_attention():
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same', input_shape=(120, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Attention())
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse', metrics=[rmse()])
    return model

# Extract features for SVM
def extract_features(data):
    features = []
    for i in range(time_step, len(data)):
        current_price = data[i]
        moving_average_5 = np.mean(data[i-5:i])
        moving_average_15 = np.mean(data[i-15:i])
        rsi = compute_rsi(data[i-14:i+1])
        features.append([current_price, moving_average_5, moving_average_15, rsi])
    return np.array(features)

def compute_rsi(data, window=14):
    delta = np.diff(data)
    gain = (delta >= 0) * delta
    loss = (delta < 0) * -delta
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Train SVM model
def train_svm_model(x_train, y_train):
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_model.fit(x_train, y_train)
    return svm_model

# Train and save the model
def train_and_save_model():
    lstm_model = create_model_with_attention()
    lstm_model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), batch_size=32, verbose=2)
    lstm_model.save(model_path)

    # Extract features for SVM training
    train_features = extract_features(train_data_initial)
    test_features = extract_features(test_data_initial)

    # Labels for SVM: 1 for buy, -1 for sell, 0 for hold (based on future price movement)
    train_labels = np.sign(np.diff(train_data_initial[time_step:], axis=0))
    test_labels = np.sign(np.diff(test_data_initial[time_step:], axis=0))

    # Train SVM model
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    svm_model = train_svm_model(train_features_scaled, train_labels)
    joblib.dump(svm_model, svm_model_path)
    joblib.dump(scaler, scaler_path)
    logging.info("Models (LSTM and SVM) trained and saved.")

# Load the models
def load_trained_model():
    return load_model(model_path, custom_objects={'Attention': Attention})

def load_trained_svm_model():
    svm_model = joblib.load(svm_model_path)
    scaler = joblib.load(scaler_path)
    return svm_model, scaler

# Function to predict next value using LSTM
def predict_next(model, data):
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
def trade_logic(lstm_model, svm_model, scaler):
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, time_step + 1)
    data = pd.DataFrame(rates).filter(['close']).values

    next_price = predict_next(lstm_model, data)
    current_price = data[-1][0]

    # Extract features for SVM
    latest_features = extract_features(data[-time_step:])
    latest_features_scaled = scaler.transform(latest_features)

    # SVM prediction: 1 for buy, -1 for sell, 0 for hold
    svm_prediction = svm_model.predict(latest_features_scaled)[-1]

    # Apply DBSCAN to detect outliers
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(latest_features_scaled)
    labels = dbscan.labels_

    # Check if current data point is an outlier
    if labels[-1] == -1:
        logging.info("Detected outlier, skipping trade.")
        return

    if svm_prediction == 1:
        request = create_request("EURUSD", mt5.ORDER_TYPE_BUY)
    elif svm_prediction == -1:
        request = create_request("EURUSD", mt5.ORDER_TYPE_SELL)
    else:
        logging.info("Hold position based on SVM prediction.")
        return

    result = mt5.order_send(request)
    logging.info(result)

# Live trading loop
def trade():
    try:
        lstm_model = load_trained_model()
        svm_model, scaler = load_trained_svm_model()
        while True:
            now = datetime.now()
            if now.minute % 15 == 0 and now.second == 0:  # Check every 15 minutes
                trade_logic(lstm_model, svm_model, scaler)
            if now.hour == 0 and now.minute == 0 and now.second == 0:  # Train the model daily at midnight
                train_and_save_model()
                lstm_model = load_trained_model()
                svm_model, scaler = load_trained_svm_model()
            time.sleep(1)  # Sleep for a second to avoid busy waiting
    except KeyboardInterrupt:
        logging.info("Trading loop interrupted by user.")
    finally:
        mt5.shutdown()

# Kick off live trading
if __name__ == "__main__":
    train_and_save_model()  # Initial training
    trade()
