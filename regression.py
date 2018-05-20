import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# Use google stock price data
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# High Low Percentage
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
# Percent Change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# adjust float for change ratio for prediction
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
# Drop data


x = np.array(df.drop(['label'], 1))
# scale the data
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

# Set classifier
clf = LinearRegression(n_jobs=-1)
# Test and Train off different data sets
clf.fit(x_train, y_train)

# Saves the classifier so you dont have to retrain
with open('lineregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle_in

clf.score(x_test, y_test)
# squared error
accuracy = clf.score(x_test, y_test)
# print(accuracy)

forecast_set = clf.predict(x_lately)

print(forecast_set, accuracy, forecast_out)

# Prepping for the graphing
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
# iterate through forecast set and day and setting those as values in the dataframe and have dates on the axis
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # Create Index if it doesnt exist, take out nan values for the forecasted values
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
