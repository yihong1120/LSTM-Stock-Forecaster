from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from stock_forecaster import ModelEvaluator, LSTMModel, DataPreprocessor, StockDataFetcher
import datetime
import os
import schedule

app = Flask(__name__)

# Train and save model
def train_model(ticker, today):
    start_date = (today - datetime.timedelta(days=10 * 365)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    look_back_candidates = [30, 60, 90, 120]

    # Fetch stock data
    stock_data = StockDataFetcher(ticker, start_date, end_date).fetch_stock_data()

    # Find the best look_back value
    best_look_back = ModelEvaluator(LSTMModel, look_back_candidates, stock_data).find_best_look_back()

    # Preprocess data with the best look_back value
    X, y, scaler = DataPreprocessor(stock_data, best_look_back).preprocess_data()

    # Train the LSTM model with the best look_back value
    model = LSTMModel.create_model((X.shape[1], 1))
    model.fit(X, y, batch_size=64, epochs=100)

    # Save the trained model
    model.save(f"models/{ticker}.h5")

# Fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

# Preprocess custom data
def preprocess_custom_data(data, train_data, look_back):
    data = data.filter(['Close'])
    scaled_data = MinMaxScaler(feature_range=(0, 1)).fit(train_data).transform(data)

    X = [scaled_data[i - look_back:i, 0] for i in range(look_back, len(scaled_data))]
    return np.reshape(X, (len(X), look_back, 1))

def delete_old_models():
    # Delete models older than 30 days
    folder_path = "models/"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.stat(file_path).st_mtime < datetime.datetime.now() - datetime.timedelta(days=30):
            os.remove(file_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    if not data or 'ticker' not in data or 'look_back' not in data or 'custom_data' not in data:
        return jsonify({"error": "Missing required fields in JSON data."}), 400

    ticker = data['ticker']
    look_back = int(data['look_back'])
    if look_back <= 0:
        return jsonify({"error": "look_back must be a positive integer."}), 400

    custom_data = data['custom_data']
    if not isinstance(custom_data, dict) or 'Close' not in custom_data:
        return jsonify({"error": "custom_data must be a dictionary containing a 'Close' key."}), 400

    # Calculate start and end dates for the past 5 years
    today = datetime.date.today()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')

    # Delete old models
    delete_old_models()

    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # Check if the DataFrame is empty
    if stock_data.empty:
        return jsonify({"error": f"Stock data for '{ticker}' not found."}), 404

    # Load the trained model or train a new one
    file_path = f"models/{ticker}.h5"
    if not os.path.exists(file_path):
        train_model(ticker, today)
    model = tf.keras.models.load_model(file_path)

    # Preprocess custom dataset
    X_custom = preprocess_custom_data(custom_data, stock_data, look_back)
    
    # Predict using the trained model
    custom_predictions = scaler.inverse_transform(model.predict(X_custom))

    # Return predictions as JSON
    return jsonify({"predictions": custom_predictions.tolist()})

if __name__ == '__main__':
    # Delete old models every day at 00:00
    schedule.every().day.at("00:00").do(delete_old_models)

    # Run Flask app
    app.run(debug=True)

