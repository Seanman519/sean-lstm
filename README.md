
# 📊 LSTM and SVM Model for Financial Time Series Prediction and Trading

This project integrates LSTM (Long Short-Term Memory) and SVM (Support Vector Machine) models for financial time series prediction and trading using TensorFlow, Keras, and MetaTrader 5. The models are trained on historical data and utilized for real-time trading decisions.

## 📝 To-Do Checklist

- [ ] **Implement ML-Based Trailing Stop-Loss**:
  - Determine appropriate trailing stop-loss size based on trade confidence.
  - Tight trailing stop for scalp trades.
  - Slower trailing stop for long swing trades.

- [ ] **Incorporate Percentage Risk-Based Lot Sizing**:
  - Implement lot sizing based on the percentage risk of the trade.
  - Adjust lot sizes dynamically based on risk assessment.

---

## 📁 Project Structure

- **Data Retrieval**: Fetches historical financial data from MetaTrader 5.
- **Data Preprocessing**: Prepares the data for LSTM training and SVM feature extraction.
- **Model Training**: 
  - Trains an LSTM model with attention mechanism for time series prediction.
  - Trains an SVM model for classification (buy, sell, hold) based on extracted features.
- **Model Evaluation**: Evaluates the LSTM model's performance during training.
- **Model Saving**: Saves the trained LSTM model and SVM model for future use.

## 🔧 Requirements

Install the required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

🚀 New Features
1. SVM Model Integration
Model: Utilizes an SVM classifier trained on extracted features (moving averages, RSI) to make trading decisions.
Integration: SVM predictions complement LSTM predictions for more robust trading strategies.

3. Real-Time Trading Loop
Live Trading: Implements a continuous trading loop based on real-time market data from MetaTrader 5.
Trading Logic: Combines LSTM price predictions with SVM classifications to execute buy, sell, or hold trading decisions.

5. Dynamic Model Updates
Daily Retraining: Automatically retrains both LSTM and SVM models daily at midnight to adapt to changing market conditions.
Model Persistence: Saves and loads trained models using joblib and Keras to ensure continuous and efficient trading operations.

📝 Usage

Initial Setup: Ensure MetaTrader 5 is installed and configured.
Environment Setup: Install required Python libraries.
Training: Run train_and_save_model() to initialize and train the LSTM and SVM models.

Live Trading: Execute trade() to start the live trading loop, making real-time trading decisions based on model predictions.

📈 Future Enhancements
Incorporate more advanced trading strategies and indicators.
Enhance real-time data processing and decision-making capabilities.
Optimize model hyperparameters dynamically based on market feedback.
