import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to fetch stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close']]

# Function to prepare data for LSTM
def prepare_data_lstm(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Function to create and train LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Function to prepare data for Prophet
def prepare_data_prophet(data):
    df = data.reset_index()
    df.columns = ['ds', 'y']
    return df

# Function to create and train Prophet model
def create_prophet_model(data):
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    return model

# Function to ensemble predictions
def ensemble_predictions(lstm_pred, prophet_pred, weights=(0.6, 0.4)):
    return weights[0] * lstm_pred + weights[1] * prophet_pred

# Main function
def main():
    # Fetch stock data
    stock_data = get_stock_data('AAPL', '2010-01-01', '2023-04-01')
    
    # Prepare data for LSTM
    X, y, scaler = prepare_data_lstm(stock_data)
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    
    # Create and train LSTM model
    lstm_model = create_lstm_model((X_train.shape[1], 1))
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)
    
    # Make LSTM predictions
    lstm_predictions = lstm_model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    
    # Prepare data for Prophet
    prophet_data = prepare_data_prophet(stock_data)
    prophet_train = prophet_data.iloc[:split+60]
    
    # Create and train Prophet model
    prophet_model = create_prophet_model(prophet_train)
    
    # Make Prophet predictions
    future_dates = prophet_model.make_future_dataframe(periods=len(X_test))
    prophet_forecast = prophet_model.predict(future_dates)
    prophet_predictions = prophet_forecast.iloc[-len(X_test):]['yhat'].values.reshape(-1, 1)
    
    # Ensemble predictions
    ensemble_pred = ensemble_predictions(lstm_predictions, prophet_predictions)
    
    # Calculate metrics
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    lstm_mse = mean_squared_error(actual_prices, lstm_predictions)
    prophet_mse = mean_squared_error(actual_prices, prophet_predictions)
    ensemble_mse = mean_squared_error(actual_prices, ensemble_pred)
    
    lstm_mae = mean_absolute_error(actual_prices, lstm_predictions)
    prophet_mae = mean_absolute_error(actual_prices, prophet_predictions)
    ensemble_mae = mean_absolute_error(actual_prices, ensemble_pred)
    
    print(f"LSTM MSE: {lstm_mse:.4f}, MAE: {lstm_mae:.4f}")
    print(f"Prophet MSE: {prophet_mse:.4f}, MAE: {prophet_mae:.4f}")
    print(f"Ensemble MSE: {ensemble_mse:.4f}, MAE: {ensemble_mae:.4f}")
    
    # Plot results
    plt.figure(figsize=(16, 8))
    plt.plot(actual_prices, label='Actual Prices')
    plt.plot(lstm_predictions, label='LSTM Predictions')
    plt.plot(prophet_predictions, label='Prophet Predictions')
    plt.plot(ensemble_pred, label='Ensemble Predictions')
    plt.legend()
    plt.title('Stock Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

if __name__ == "__main__":
    main()
