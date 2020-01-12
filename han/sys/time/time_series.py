import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Model_Data.xlsx")
df.head()

# extract time series
time_series_data = df["Active_Rig_Count"]

# plot
plt.plot(time_series_data.values)
plt.show()

# ============================ ARIMA ============================

from statsmodels.tsa.arima_model import ARIMA

X_train = list(time_series_data.values)
predictions = []

# forecast the next 200 data points
for _ in range(200):
    model = ARIMA(X_train, order=(5,1,0))
    model_fit = model.fit(disp=False)
    output = model_fit.forecast()
    y_hat = output[0] # new forecasted data point
    predictions.append(y_hat)
    X_train.append(y_hat) # add forecasted data point back to X_train to train model in the next round

# plot
plt.plot(range(len(time_series_data.values)), time_series_data.values, color='blue', label='History data')
plt.plot(range(len(time_series_data.values), len(time_series_data.values) + len(predictions)), predictions, color='red', label='Forecasted data')
plt.legend()
plt.show()

# ============================ LSTM ============================

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

look_back = 200 # look back 40 data points (months) to make prediction

# convert an array of values into a dataset matrix
def create_dataset(data, look_back=1):
	dataX, dataY = [], []
	for i in range(len(data)-look_back-1):
		a = data[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(data[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

data = np.reshape(time_series_data.values, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
X, y = create_dataset(data, look_back)
# reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# create model and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=50, batch_size=1, verbose=2)

# make prediction
val = data[-look_back:] # last batch in the training data
predictions = []

# forecast the next 200 data points
for _ in range(200):
    pred = model.predict(val.reshape(1,1,look_back))
    predictions = np.append(predictions,pred)
    val = np.append(np.delete(val, 0), pred) # update the data batch to be fed in the next iteration
predictions = predictions.reshape(predictions.shape[0],1)

# reverse forecasted results to its original scale
predictions = scaler.inverse_transform(predictions)

# plot
data_len = len(data)
plt.plot(list(range(data_len)),time_series_data.values,label="actual data")
plt.plot(list(range(data_len,data_len+len(predictions))),predictions,label="predicted data")
plt.legend()
plt.show()