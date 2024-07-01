import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from tqdm import tqdm
import vectorbt as vbt


if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()


SYMBOL = "GBPUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
SEQ_LENGTH = 50
INPUT_SIZE = 1
HIDDEN_SIZE = 50
OUTPUT_SIZE = 1
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 250
HOLD_THRESHOLD = 0.01 


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.context_vector = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(self, lstm_output):
        scores = self.context_vector(torch.tanh(self.attention(lstm_output)))
        attention_weights = torch.softmax(scores, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        lstm_output, _ = self.lstm(x, (h0, c0))
        context, attention_weights = self.attention(lstm_output)
        out = self.fc(context)
        return out, attention_weights


def fetch_data(symbol, timeframe, n=120):
    utc_from = datetime.now() - timedelta(days=n)
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, datetime.now())
    rates_frame = pd.DataFrame(rates)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    return rates_frame


def preprocess_data(data):
    data['close'] = data['close'].astype(float)
    data['scaled_close'] = (data['close'] - data['close'].mean()) / data['close'].std()
    return data


def create_sequences(data, seq_length=50):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data['scaled_close'].values[i:i+seq_length])
        labels.append(data['scaled_close'].values[i+seq_length])
    return np.array(sequences), np.array(labels)


def train_model(X_train, y_train):
    model = BiLSTMWithAttention(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for _ in tqdm(range(1), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False):
            outputs, _ = model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}')
    
    return model


def predict_next(data, model, seq_length=50, hold_threshold=HOLD_THRESHOLD):
    mean = data['close'].mean()
    std = data['close'].std()
    model.eval()
    with torch.no_grad():
        seq = torch.tensor(data['scaled_close'].values[-seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        pred, _ = model(seq)
    scaled_prediction = pred.item()
    prediction = (scaled_prediction * std) + mean  # De-normalize the prediction
    
    signal = 'Hold'
    if abs(scaled_prediction) > hold_threshold:
        signal = 'Buy' if scaled_prediction > 0 else 'Sell'

    return prediction, signal


data = fetch_data(SYMBOL, TIMEFRAME)
data = preprocess_data(data)
X, y = create_sequences(data, SEQ_LENGTH)


train_size = int(len(X) * 0.8)
X_train = torch.tensor(X[:train_size], dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y[:train_size], dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X[train_size:], dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y[train_size:], dtype=torch.float32).unsqueeze(-1)

model = train_model(X_train, y_train)


model.eval()
with torch.no_grad():
    predictions, _ = model(X_test)
    predictions = predictions.squeeze().numpy()

mean = data['close'].mean()
std = data['close'].std()
actuals = y_test.squeeze().numpy()
predictions = (predictions * std) + mean  # De-normalize the predictions
actuals = (actuals * std) + mean  # De-normalize the actual values
dates = data['time'].iloc[train_size + SEQ_LENGTH:].values


signals = pd.Series(index=dates, data=np.nan)
for i, (pred, date) in enumerate(zip(predictions, dates)):
    signal = np.nan
    if abs(pred - actuals[i]) > HOLD_THRESHOLD * std:
        signal = 1 if pred > actuals[i] else -1
    signals.loc[date] = signal


close_prices = data['close'].iloc[train_size + SEQ_LENGTH:]
close_prices.index = signals.index

entries = signals == 1
exits = signals == -1

portfolio = vbt.Portfolio.from_signals(close_prices, entries, exits, freq='15min')
performance = portfolio.stats()
print(performance)

mt5.shutdown()
