import torch
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import time
import logging
import requests
from transformers import pipeline


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if not mt5.initialize():
    logging.error("initialize() failed")
    mt5.shutdown()

# Configuration
SYMBOL = "USDJPY"
TIMEFRAME = mt5.TIMEFRAME_M15
SEQ_LENGTH = 50
INPUT_SIZE = 1
HIDDEN_SIZE = 50
OUTPUT_SIZE = 1
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 180
LOT_SIZE = 0.1  
PREDICTION_INTERVAL = 900  
MODEL_PATH = 'bilstm_modelnews.pth'
NEWS_API_KEY = 'fe593204b9d045cf9fc5043ffa289129'
NEWS_ENDPOINT = 'https://newsapi.org/v2/everything'
CONFIDENCE_THRESHOLD = 0.8  


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size * 2)  # Batch normalization
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.bn(out[:, -1, :])
        out = self.fc(out)
        return out


def fetch_data(symbol, timeframe, n=120):
    utc_from = datetime.now() - timedelta(days=n)
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, datetime.now())
    if rates is None:
        logging.error("Failed to fetch data")
        return None
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


def train_model():
    data = fetch_data(SYMBOL, TIMEFRAME)
    if data is None:
        logging.error("No data fetched. Exiting training.")
        return None
    data = preprocess_data(data)
    X, y = create_sequences(data, SEQ_LENGTH)

    X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    model = BiLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    for epoch in range(NUM_EPOCHS):
        model.train()
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        scheduler.step(loss)
        
        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')
    
   
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
    return model


model = train_model()


def load_model(model_path):
    model = BiLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model


def execute_order(symbol, order_type, lot_size):
    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    deviation = 20
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": price - 100 * point if order_type == mt5.ORDER_TYPE_BUY else price + 100 * point,
        "tp": price + 100 * point if order_type == mt5.ORDER_TYPE_BUY else price - 100 * point,
        "deviation": deviation,
        "magic": 234000,
        "comment": "LSTM-15",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Order failed, retcode={result.retcode}")
    return result


def predict_and_trade(data, model, seq_length=50, lot_size=0.1, sentiment_score=0):
    mean = data['close'].mean()
    std = data['close'].std()
    model.eval()
    with torch.no_grad():
        seq = torch.tensor(data['scaled_close'].values[-seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        pred = model(seq)
    scaled_prediction = pred.item()
    prediction = (scaled_prediction * std) + mean  # De-normalize the prediction

   
    confidence = abs(scaled_prediction)

    if confidence < CONFIDENCE_THRESHOLD:
        logging.info(f"Confidence ({confidence:.2f}) is below the threshold ({CONFIDENCE_THRESHOLD:.2f}). Holding position.")
        return

    if sentiment_score > 0:
        logging.info("Positive sentiment detected, considering a Buy signal.")
        signal = 'Buy'
        order_type = mt5.ORDER_TYPE_BUY
    elif sentiment_score < 0:
        logging.info("Negative sentiment detected, considering a Sell signal.")
        signal = 'Sell'
        order_type = mt5.ORDER_TYPE_SELL
    else:
        logging.info("Neutral sentiment detected, considering a Hold signal.")
        signal = 'Hold'
        order_type = None  

    logging.info(f'Next predicted closing price: {prediction:.5f}')
    logging.info(f'Signal: {signal}')
    logging.info(f'Confidence: {confidence:.2f}')

    if order_type is not None:
        # Execute the order
        result = execute_order(SYMBOL, order_type, lot_size)
        logging.info(result)
    else:
        logging.info("No trade executed, holding position.")


def fetch_news_and_analyze_sentiment():
    params = {
        'q': 'AUDUSD',
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 5
    }
    response = requests.get(NEWS_ENDPOINT, params=params)
    if response.status_code != 200:
        logging.error("Failed to fetch news data")
        return 0
    
    news_data = response.json()
    articles = news_data.get('articles', [])
    
    sentiment_analyzer = pipeline('sentiment-analysis')
    sentiments = [sentiment_analyzer(article['title'])[0]['label'] for article in articles]
    
    positive_sentiment_count = sentiments.count('POSITIVE')
    negative_sentiment_count = sentiments.count('NEGATIVE')
    
    sentiment_score = positive_sentiment_count - negative_sentiment_count
    logging.info(f"Sentiment score: {sentiment_score}")
    
    return sentiment_score


model = load_model(MODEL_PATH)


while True:
    data = fetch_data(SYMBOL, TIMEFRAME)
    if data is not None:
        data = preprocess_data(data)
        sentiment_score = fetch_news_and_analyze_sentiment()
        predict_and_trade(data, model, SEQ_LENGTH, LOT_SIZE, sentiment_score)
    time.sleep(PREDICTION_INTERVAL)
