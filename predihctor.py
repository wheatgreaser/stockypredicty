# predihctor.py
import torch
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import model as md  

input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
seq_length = 10

TICKER = "AAPL"  # Change to the stock you want to predict for
CSV_FILE = "close_prices.csv"

def update_close_prices():
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1y")
    df = df[["Close"]]
    df.to_csv(CSV_FILE)
    return df

def run_prediction():
    # Always refresh data before prediction
    df = update_close_prices()

    df = df.dropna(subset=["Close"])
    close_prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaler.fit(close_prices)

    close_prices_scaled = scaler.transform(close_prices).flatten()

    input_seq = close_prices_scaled[-seq_length:] 
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    model = md.LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load("lstm_model.pth"))
    model.eval()

    with torch.no_grad():
        pred_scaled = model(input_tensor).item()

    pred_price = scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]
    return round(pred_price, 3)
