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
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MetaTrader5 Initialization
if not mt5.initialize():
    logging.error("initialize() failed")
    mt5.shutdown()

# Configuration
SYMBOL = "USDJPY"
TIMEFRAMES = [mt5.TIMEFRAME_H4, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M1]
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
MAX_DRAWDOWN = 0.03

# Custom actor-critic policy with BiLSTM integration
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True, bidirectional=True)
        self.fc_actor = nn.Linear(HIDDEN_SIZE * 2, self.action_space.n)
        self.fc_critic = nn.Linear(HIDDEN_SIZE * 2, 1)

    def forward(self, obs, deterministic=False):
        # Extract features using the LSTM
        h0 = torch.zeros(self.num_layers * 2, obs.size(0), HIDDEN_SIZE).to(obs.device)
        c0 = torch.zeros(self.num_layers * 2, obs.size(0), HIDDEN_SIZE).to(obs.device)
        lstm_out, _ = self.lstm(obs, (h0, c0))
        
        actor_features = lstm_out[:, -1, :]
        critic_features = lstm_out[:, -1, :]

        logits = self.fc_actor(actor_features)
        values = self.fc_critic(critic_features)

        return logits, values


# Environment setup for SMC/ICT market strategies with multiple timeframes and drawdown control
class MarketEnv(gym.Env):
    def __init__(self, symbol, timeframes, seq_length=SEQ_LENGTH, max_drawdown=MAX_DRAWDOWN):
        super(MarketEnv, self).__init__()
        self.symbol = symbol
        self.timeframes = timeframes
        self.seq_length = seq_length
        self.max_drawdown = max_drawdown
        self.initial_balance = 10000  
        self.current_balance = self.initial_balance
        self.highest_balance = self.initial_balance
        self.llm_usage_threshold = 0.05  # Threshold for volatility to trigger LLM

        # Action space: Buy, Sell, Hold
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price data sequence for each timeframe
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(timeframes), seq_length, INPUT_SIZE), dtype=np.float32)
        
        # Fetch initial market data for all timeframes
        self.data = {tf: fetch_data(self.symbol, tf) for tf in self.timeframes}
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.data = {tf: fetch_data(self.symbol, tf) for tf in self.timeframes}
        self.current_balance = self.initial_balance
        self.highest_balance = self.initial_balance
        return self._next_observation()

    def _next_observation(self):
        # Combine data from multiple timeframes into a single observation
        obs = []
        for tf in self.timeframes:
            obs.append(self.data[tf]['scaled_close'].values[self.current_step:self.current_step + self.seq_length])
        return np.array(obs).reshape(len(self.timeframes), self.seq_length, 1)

    def step(self, action):
        reward = self._calculate_reward(action)
        self.current_step += 1
        
        done = self.current_step + self.seq_length >= len(self.data[self.timeframes[-1]])
        obs = self._next_observation()

        # Check for drawdown limit
        drawdown = (self.highest_balance - self.current_balance) / self.highest_balance
        if drawdown > self.max_drawdown:
            done = True
            reward -= 10  # Penalty for exceeding drawdown

        # Check for volatility and decide to use LLM or not
        if self._check_volatility():
            self._use_llm()

        return obs, reward, done, {}

    def _calculate_reward(self, action):
        reward = 0
        current_price = self.data[mt5.TIMEFRAME_M15]['close'].values[self.current_step + self.seq_length - 1]
        next_price = self.data[mt5.TIMEFRAME_M15]['close'].values[self.current_step + self.seq_length]
        
        if action == 0:  # Buy
            reward = next_price - current_price
        elif action == 1:  # Sell
            reward = current_price - next_price
        elif action == 2:  # Hold
            reward = -abs(next_price - current_price) * 0.1  # Small penalty for holding

        # Simulate balance changes based on reward
        self.current_balance += reward
        self.highest_balance = max(self.highest_balance, self.current_balance)
        
        return reward

    def _check_volatility(self):
        """Check if market volatility exceeds a threshold, triggering the LLM."""
        current_price = self.data[mt5.TIMEFRAME_M15]['close'].values[self.current_step + self.seq_length - 1]
        previous_price = self.data[mt5.TIMEFRAME_M15]['close'].values[self.current_step + self.seq_length - 2]
        price_change = abs(current_price - previous_price) / previous_price

        return price_change > self.llm_usage_threshold

    def _use_llm(self):
        """Use the LLM to analyze news and adjust trading signals if necessary."""
        sentiment_score = fetch_news_and_analyze_sentiment()
        if sentiment_score > 0:
            logging.info("LLM suggests bullish sentiment. Adjusting to Buy bias.")
        elif sentiment_score < 0:
            logging.info("LLM suggests bearish sentiment. Adjusting to Sell bias.")
        else:
            logging.info("LLM indicates neutral sentiment. No adjustment.")

        # Modify reward or trading bias based on sentiment -> Must Do you Moron!!!!!!


# BiLSTM Model for market prediction
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_size)

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
    data = fetch_data(SYMBOL, mt5.TIMEFRAME_M15)
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


def load_model(model_path):
    model = BiLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model

# Predict and trade using RL and BiLSTM
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
        "comment": "RL-SMC-ICT",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Order failed, retcode={result.retcode}")
    return result

# Fetch news sentiment using transformer-based LLM
def fetch_news_and_analyze_sentiment():
    params = {
        'q': 'USDJPY',
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

# Main loop for running the model
def run_model_loop():
    model = load_model(MODEL_PATH)
    while True:
        data = fetch_data(SYMBOL, mt5.TIMEFRAME_M15)
        if data is not None:
            data = preprocess_data(data)
            sentiment_score = fetch_news_and_analyze_sentiment()
            predict_and_trade(data, model, SEQ_LENGTH, LOT_SIZE, sentiment_score)
        time.sleep(PREDICTION_INTERVAL)

# RL Training function with PPO
def train_rl_model():
    env = MarketEnv(SYMBOL, TIMEFRAMES)
    model = PPO(CustomActorCriticPolicy, env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_smc_ict")
    
    return model

# Train RL model
rl_model = train_rl_model()

# Start trading loop with trained model
run_model_loop()
