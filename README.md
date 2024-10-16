# Reinforcement Learning for Financial Markets using BiLSTM, LLM and PPO

This project implements a trading algorithm that leverages **Reinforcement Learning (RL)**, **BiLSTM neural networks**, and **transformer-based sentiment analysis** for **financial market predictions**. It integrates with MetaTrader5 for real-time trading and uses market data across multiple timeframes. The system is capable of analyzing market volatility and news sentiment, adjusting trading strategies accordingly.

> **Note:** This codebase is under active development and maintenance. New features and improvements are regularly added, and more insights will be shared as the research progresses.

## Features

- **Reinforcement Learning (RL)** using **Proximal Policy Optimization (PPO)**.
- Custom **Actor-Critic policy** with **BiLSTM** for market prediction.
- **MetaTrader5 integration** for real-time market data and trade execution.
- Sentiment analysis using **transformer-based language models** to incorporate news data into trading decisions.
- Multi-timeframe analysis with support for **SMC/ICT** market strategies.
- Built-in risk management through **maximum drawdown limits** and dynamic decision-making.

## Project Structure

- **`anita.py`**: Main script that runs the entire trading system, combining market prediction, RL strategies, and trading execution.
- **`bilstm_modelnews.pth`**: Pre-trained BiLSTM model for market predictions.
- **`MarketEnv`**: Custom environment class for simulating financial markets with multi-timeframe data.
- **`BiLSTM`**: Neural network model for predicting market trends.
- **`CustomActorCriticPolicy`**: Policy for RL training using PPO.
- **`fetch_data`, `preprocess_data`, `fetch_news_and_analyze_sentiment`**: Utility functions for data retrieval, processing, and sentiment analysis.


To run the system, simply execute the `anita.py` script:

```bash
python anita.py
```

## Support

If you find this project helpful and would like to support the continued development and maintenance, you can [**Buy Me a Coffee**](https://www.buymeacoffee.com/seany519). Your support will help fund additional research and future updates.




