import yfinance as yf

ticker = yf.Ticker("AAPL")
data = ticker.history(period="1y")
print(data[["Close"]])
data[["Close"]].to_csv("close_prices.csv")
