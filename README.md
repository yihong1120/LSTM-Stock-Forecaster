# LSTM-Stock-Forecaster

LSTM-Stock-Forecaster is a deep learning-based stock price prediction application using Long Short-Term Memory (LSTM) neural networks and a Flask API. The application fetches historical stock data and trains an LSTM model to predict stock closing prices. Users can interact with the Flask API to receive predictions for their custom data.

## Features

- Fetches historical stock data using the yfinance API
- Data preprocessing for neural network input
- LSTM model creation and training
- Model evaluation to find the best look_back parameter
- Flask API for receiving custom data and returning predictions
- Automatic deletion of old models

## Installation

1. Clone the repository:

    git clone https://github.com/yourusername/LSTM-Stock-Forecaster.git
    cd LSTM-Stock-Forecaster

2. Install the required packages:

    pip install -r requirements.txt

## Usage

1. Run the Flask application:

    python app.py

2. Send a POST request to the Flask API containing the required JSON data:

{
"ticker": "AAPL",
"look_back": 60,
"custom_data": {
"Close": [
150.0,
152.3,
154.1,
...
]
}
}

3. The API will return the stock price predictions in JSON format:

{
"predictions": [
[152.33],
[153.41],
...
]
}

## Contributing

1. Fork the repository on GitHub
2. Create a feature branch: `git checkout -b my-feature-branch`
3. Commit your changes: `git commit -am 'Add my feature'`
4. Push the branch to GitHub: `git push origin my-feature-branch`
5. Create a pull request against the master branch

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
