# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load your COVID vaccine analysis dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('country_vaccinations.csv')
data = pd.read_csv('country_vaccinations_by_manufacturer.csv')

# Assuming 'date' is the column containing the date and 'vaccine_doses' is the column with the number of vaccine doses
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
ts_data = data['vaccine_doses']

# Visualize the time series data
plt.figure(figsize=(12, 6))
plt.plot(ts_data)
plt.title('COVID Vaccine Doses Time Series')
plt.xlabel('Date')
plt.ylabel('Number of Vaccine Doses')
plt.show()

# Check stationarity (perform Dickey-Fuller test)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    print('Results of Dickey-Fuller Test:')
    print('Test Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'{key}: {value}')

test_stationarity(ts_data)

# Make the time series stationary by differencing
ts_data_diff = ts_data.diff().dropna()

# Check stationarity again after differencing
test_stationarity(ts_data_diff)

# Build and train the ARIMA model
model = ARIMA(ts_data_diff, order=(5, 1, 0))  # ARIMA(p, d, q) order selection based on analysis
model_fit = model.fit(disp=0)

# Forecast future values
forecast_steps = 10  # Number of steps to forecast into the future
forecast, stderr, conf_int = model_fit.forecast(steps=forecast_steps)

# Calculate lower and upper confidence intervals
lower_conf_int = forecast - 1.96 * stderr
upper_conf_int = forecast + 1.96 * stderr

# Convert differenced values back to the original scale if differencing was applied
forecast = forecast.cumsum()
lower_conf_int = lower_conf_int.cumsum()
upper_conf_int = upper_conf_int.cumsum()

# Plot the original time series, forecast, and confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data, label='Observed')
plt.plot(pd.date_range(ts_data.index[-1], periods=forecast_steps, closed='right'), forecast, label='Forecast')
plt.fill_between(pd.date_range(ts_data.index[-1], periods=forecast_steps, closed='right'), lower_conf_int, upper_conf_int, color='pink', alpha=0.3, label='95% Confidence Interval')
plt.title('COVID Vaccine Doses Forecasting with ARIMA')
plt.xlabel('Date')
plt.ylabel('Number of Vaccine Doses')
plt.legend()
plt.show()

# Evaluate the model
actual_values = ts_data[-forecast_steps:]
rmse = sqrt(mean_squared_error(actual_values, forecast))
print('Root Mean Squared Error (RMSE):', rmse)
