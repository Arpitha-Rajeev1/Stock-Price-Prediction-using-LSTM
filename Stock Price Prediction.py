import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Define the date range and stock symbol
start = '2012-01-01'
end = '2022-12-21'
stock = 'GOOG'

# Download stock data using Yahoo Finance
data = yf.download(stock, start, end)

# Reset index to make 'Date' a column
data.reset_index(inplace=True)

# Calculate 100-day moving average
ma_100_days = data.Close.rolling(100).mean()

# Plotting 100-day moving average and closing price
plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.show()

# Calculate 200-day moving average
ma_200_days = data.Close.rolling(200).mean()

# Plotting 100-day and 200-day moving averages with closing price
plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')

# Remove NaN values from the data
data.dropna(inplace=True)

# Prepare training data and test data
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

# Scale the training data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_train_scale = scaler.fit_transform(data_train)

# Initialize empty arrays for features (x) and target (y)
x = []
y = []

# Create sequences for LSTM input
for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i])
    y.append(data_train_scale[i,0])

# Convert arrays to numpy arrays
x, y = np.array(x), np.array(y)

# Import necessary Keras modules for LSTM model creation
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# Build LSTM model
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
              input_shape = ((x.shape[1],1))))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation='relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer = 'adam', loss='mean_squared_error')

model.fit(x,y, epochs = 50, batch_size = 32, verbose = 1)

model.summary()

# Prepare test data and make predictions
pas_100_days = data_train.tail(100)

data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

data_test_scale = scaler.fit_transform(data_test)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x, y = np.array(x), np.array(y)

y_predict = model.predict(x)

scale = 1/scaler.scale_

y_predict = y_predict*scale

y = y*scale

# Plotting Predicted vs Original Prices
plt.figure(figsize=(10,8))
plt.plot(y_predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

model.save('Stock Predictions Model.keras')

# Calculate various error metrics for model evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y, y_predict))
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y - y_predict) / y)) * 100
print("Mean Absolute Percentage Error (MAPE):", mape)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y, y_predict)
print("Mean Squared Error (MSE):", mse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y, y_predict)
print("Mean Absolute Error (MAE):", mae)

import matplotlib.pyplot as plt

# Calculated error values
error_metrics = ['MAE', 'RMSE', 'MAPE', 'MSE']
error_values = [mae, rmse, mape, mse]

# Plotting the error measures
plt.figure(figsize=(8, 6))
plt.bar(error_metrics, error_values, color='skyblue')
plt.title('Error Measures Comparison')
plt.xlabel('Error Metrics')
plt.ylabel('Error Values')
plt.ylim(0, max(error_values) * 1.2)  # Set y-axis limit for better visualization
plt.xticks(rotation=45)

for i, value in enumerate(error_values):
    plt.text(i, value + 1, str(round(value, 2)), ha='center', va='bottom', fontsize=10)
plt.show()

if y_predict[-1] > y_predict[0]:
    print("The model predicts an upward trend.")
    if y[-1] > y[0]:
        print("The actual prices also show an upward trend. Consider holding or buying.")
    else:
        print("Actual prices do not align with the predicted upward trend. Further analysis is advised.")
else:
    print("The model predicts a downward or stagnant trend.")
    if y[-1] < y[0]:
        print("The actual prices also display a downward trend. Consider selling or avoiding.")
    else:
        print("Actual prices do not align with the predicted downward or stagnant trend. Further analysis is advised.")
