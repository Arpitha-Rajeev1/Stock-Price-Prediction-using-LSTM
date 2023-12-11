import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load the pre-trained LSTM model
model = load_model('Stock Predictions Model.keras')

# Set up Streamlit interface for stock price prediction
st.header('Stock Price Predictor')

# Collect user input for stock symbol, start and end dates
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download stock data using yfinance
data = yf.download(stock, start, end)

# Display retrieved stock data
st.subheader('Stock Data (2012-2022)')
st.write(data)

# Prepare data for analysis - split into train and test sets
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

# Scale the test data using MinMaxScaler for LSTM
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# Tail of training data for continuity in test set
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plotting Price vs Time Chart with 50 days Moving Average
st.subheader('Price vs Time Chart with 50 days MA')  # Added comment
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='50-day MA')  
plt.plot(data.Close, 'g', label='Closing Price')  
plt.legend()  
plt.show()
st.pyplot(fig1)

# Plotting Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='50-day MA') 
plt.plot(ma_100_days, 'b', label='100-day MA')  
plt.plot(data.Close, 'g', label='Closing Price')  
plt.legend()
plt.show()
st.pyplot(fig2)

# Plotting Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label='100-day MA')  
plt.plot(ma_200_days, 'b', label='200-day MA')  
plt.plot(data.Close, 'g', label='Closing Price')  
plt.legend()
plt.show()
st.pyplot(fig3)

# Prepare data for LSTM prediction
x = []
y = []

# Prepare data sequences for prediction
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

# Predict using the loaded LSTM model
predict = model.predict(x)

# Rescale predicted and actual prices
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

# Plotting Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
