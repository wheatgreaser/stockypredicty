# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from predihctor import run_prediction, update_close_prices
from apscheduler.schedulers.background import BackgroundScheduler

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scheduler = BackgroundScheduler()
scheduler.add_job(update_close_prices, "interval", days=1)
scheduler.start()

@app.get("/stocks")
def get_stock_data():
    ticker_symbols = ["AAPL", "NVDA", "GOOG", "AMZN", "META"]
    results = {}

    for symbol in ticker_symbols:
        t = yf.Ticker(symbol)
        data = t.history(period="1d")
        results[symbol] = data["Close"].iloc[0] 

    return results
@app.get("/predict")

def predict():
    prediction = run_prediction()
    return {"Predicted Next Day Close Price": prediction}