# -Stock-Market-Analysis-using-Python
Fetch live data from Yahoo Finance - Candlestick chart (Plotly) - Moving Averages (20, 50 days) - Simple Linear Regression prediction - Dynamic input for any stock

!pip install yfinance pandas matplotlib plotly scikit-learn seaborn

#IMPORTING USEFUL LIBRARIES

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime

#ACCESS THE STOCK MARKET DATA

stock = yf.Ticker("TSLA")
df = stock.history(period="1y")  # or "1y", "5d", etc.
df.reset_index(inplace=True)
df.head(365)

#SUMMARY OF THE DATA

print(df.describe())  #provides a statistical summary of a DataFrame or Series
# print(df.info())      # provides a concise summary of a DataFrame

#PLOTTING DATE VS PRICE CHART

plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Close'], label='Close Price')
# plt.plot(df['Date'], df['Close'])
plt.title('Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
# plt.grid(True)
plt.show()

#CANDLESTICK REPRESENTATION

fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
fig.update_layout(title='Candlestick Chart', xaxis_rangeslider_visible=False)
fig.show()

#MOVING AVERAGES(.rolling method())

df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.plot(df['Date'], df['MA20'], label='20-Day MA')
plt.plot(df['Date'], df['MA50'], label='50-Day MA')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Moving Averages')
plt.legend()
plt.show()

#Predict using LINEAR REGRESSION

df['Date_ordinal'] = pd.to_datetime(df['Date']).map(datetime.toordinal)

X = df[['Date_ordinal']]
y = df['Close']

model = LinearRegression()
model.fit(X, y)

df['Predicted'] = model.predict(X)

plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Close'], label='Actual', color='red')
plt.plot(df['Date'], df['Predicted'], label='Predicted', linestyle=':',color='purple')
plt.title('Linear Regression Prediction')
plt.legend()
plt.show()\
#AUTOMATE

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    df.reset_index(inplace=True)
    return df

data = get_stock_data("TSLA")
