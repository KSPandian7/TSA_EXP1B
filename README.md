### DEVELOPED BY : KULASEKARAPANDIAN K
### REGISTER NO : 212222240052
### Date: 

# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA


## AIM:
To perform regular differencing, seasonal adjustment, and log transformation on cryptocurrency data (Ethereum) to convert non-stationary data to stationary.

## ALGORITHM:
1. Import the required packages like pandas, numpy, and matplotlib.
2. Read the data using pandas.
3. Perform data preprocessing if needed, and apply regular differencing, seasonal adjustment, and log transformation.
4. Plot the data before and after applying regular differencing, seasonal adjustment, and log transformation.
5. Display the overall results.
6. 
## PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the Ethereum data
file_path = 'ETH15M.csv'
data = pd.read_csv(file_path, parse_dates=['dateTime'], index_col='dateTime')

# Visualize the original data (Closing price)
plt.figure(figsize=(10, 6))
plt.plot(data['close'], label='Original Data')
plt.title('Original Time Series (Ethereum Closing Price)')
plt.xlabel('DateTime')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Perform Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

# ADF test on original data
print("Original Data ADF Test:")
adf_test(data['close'])

# Differencing to make the data stationary
data_diff = data['close'].diff().dropna()

# ADF test on differenced data
print("\nDifferenced Data ADF Test:")
adf_test(data_diff)

# Visualize the differenced data
plt.figure(figsize=(10, 6))
plt.plot(data_diff, label='Differenced Data')
plt.title('Differenced Time Series (Ethereum Closing Price)')
plt.xlabel('DateTime')
plt.ylabel('Differenced Closing Price')
plt.legend()
plt.show()

# Log Transformation
data_log = np.log(data['close'])

# ADF test on log-transformed data
print("\nLog-Transformed Data ADF Test:")
adf_test(data_log)

# Visualize the log-transformed data
plt.figure(figsize=(10, 6))
plt.plot(data_log, label='Log-Transformed Data')
plt.title('Log-Transformed Time Series (Ethereum Closing Price)')
plt.xlabel('DateTime')
plt.ylabel('Log of Closing Price')
plt.legend()
plt.show()

# SEASONAL ADJUSTMENT using seasonal decomposition
result = seasonal_decompose(data['close'], model='additive', period=12)  # assuming 12-step seasonality
seasonally_adjusted = data['close'] - result.seasonal

# Visualize the seasonal component and the seasonally adjusted data
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(result.seasonal, label='Seasonal Component')
plt.title('Seasonal Component (Ethereum Closing Price)')
plt.xlabel('DateTime')
plt.ylabel('Seasonal Component')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(seasonally_adjusted, label='Seasonally Adjusted Data')
plt.title('Seasonally Adjusted Time Series (Ethereum Closing Price)')
plt.xlabel('DateTime')
plt.ylabel('Seasonally Adjusted Closing Price')
plt.legend()

plt.tight_layout()
plt.show()

```

## OUTPUT:

### REGULAR DIFFERENCING:
![image](https://github.com/user-attachments/assets/c5f8a3b2-0bee-456e-bd33-e731356d27cb)


### SEASONAL ADJUSTMENT:
![image](https://github.com/user-attachments/assets/62430a2d-cb78-4732-84ff-e028d872b653)



### LOG TRANSFORMATION:
![image](https://github.com/user-attachments/assets/70eb1d8f-e365-43f1-8f6a-13fc66be08ee)



## RESULT:
Thus, we have created the Python code for the conversion of non-stationary to stationary data using regular differencing, seasonal adjustment, and log transformation on cryptocurrency (Ethereum) data.
