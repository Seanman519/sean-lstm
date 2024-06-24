
# 📊 LSTM and SVM Model for Financial Time Series Prediction and Trading

This project integrates LSTM (Long Short-Term Memory) and SVM (Support Vector Machine) models for financial time series prediction and trading using TensorFlow, Keras, and MetaTrader 5. The models are trained on historical data and utilized for real-time trading decisions.



## 🔧 Requirements

Install the required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

📈 Features:

* Dynamic Trailing Stop-Loss: Adjust stop-loss dynamically based on trade confidence.

* Percentage Risk-Based Lot Sizing: Implement lot sizing based on the percentage risk of the trade.

* Rollover Protection: Prohibit trades during rollover periods (11pm to 1am).

* FTMO Rules Compliance: Check and enforce FTMO trading rules, including daily loss limits and maximum drawdown.

* Historical Data Retrieval: Retrieve historical data from MetaTrader5 platform for analysis and model training.

* Data Preparation: Prepare historical data for model training by scaling and structuring into suitable sequences.

* Model Creation: Construct predictive models using Deep Learning techniques like Convolutional Neural Networks (CNNs) and Long Short-Term Memory Networks (LSTMs).

* Model Training: Train models with early stopping criteria to optimize performance and prevent overfitting.

* Model Loading: Load pre-trained models for real-time prediction and decision-making.

* Prediction Generation: Generate price predictions, confidence levels, and trailing stop values based on current market data.

* Trade Execution: Automatically execute trades based on predictive signals, dynamically adjusting take-profit and stop-loss levels.

* Risk Management: Calculate trade volume based on predefined risk percentage per trade and adjust trade size to meet broker's minimum requirements.

* Live Trading Loop: Continuously monitor market conditions and execute trades at specified intervals.

* Interrupt Handling: Handle user interruptions gracefully during live trading sessions.




