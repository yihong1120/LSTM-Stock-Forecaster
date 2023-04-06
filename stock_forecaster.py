import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import datetime, sys


class StockDataFetcher:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_stock_data(self):
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if stock_data.empty:
            print(f"Error: Stock data for '{self.ticker}' not found.")
            sys.exit(1)
        return stock_data


class DataPreprocessor:
    def __init__(self, data, look_back=60):
        self.data = data
        self.look_back = look_back

    def preprocess_data(self):
        if self.data.empty:
            print(f"Error: Data is empty.")
            sys.exit(1)

        data = self.data.filter(['Close'])
        dataset = data.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i - self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, scaler


class LSTMModel:
    @staticmethod
    def create_model(input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model


class ModelEvaluator:
    def __init__(self, model, look_back_candidates, stock_data):
        self.model = model
        self.look_back_candidates = look_back_candidates
        self.stock_data = stock_data

    def train_and_evaluate_model(self, look_back):
        X, y, scaler = DataPreprocessor(self.stock_data, look_back).preprocess_data()
        tscv = TimeSeriesSplit(n_splits=5)

        mse_scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = self.model.create_model((X_train.shape[1], 1))
            model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=0)

            predictions = model.predict(X_test)
            mse = tf.keras.losses.mean_squared_error(y_test, predictions)
            mse_scores.append(np.mean(mse))

        return np.mean(mse_scores)

    def find_best_look_back(self):
        mse_scores = []
        for look_back in self.look_back_candidates:
            mse = self.train_and_evaluate_model(look_back)
            print(f"Mean Squared Error for look_back {look_back}: {mse}")
            mse_scores.append(mse)

        best_look_back = self.look_back_candidates[np.argmin(mse_scores)]
        print(f"Best look_back: {best_look_back}")
        return best_look_back

if __name__ == "__main__":
    today = datetime.date.today()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - datetime.timedelta(days=10 * 365)).strftime('%Y-%m-%d')
    ticker = 'AAPL'
    stock_data = StockDataFetcher(ticker, start_date, end_date).fetch_stock_data()
    # Check if the DataFrame is empty
    if stock_data.empty:
        print(f"Error: Stock data for '{ticker}' not found.")
        sys.exit(1)  # Exit the program with an error code

    look_back_candidates = [30, 60, 90, 120]
    lstm_model = LSTMModel()
    model_evaluator = ModelEvaluator(lstm_model, look_back_candidates, stock_data)
    best_look_back = model_evaluator.find_best_look_back()

    X, y, scaler = DataPreprocessor(stock_data, best_look_back).preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lstm_model.create_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=64, epochs=100)
    model.save(f"{ticker}.h5")

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
