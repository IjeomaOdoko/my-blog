# Create Stock Price Indicator Dashboard in Power BI


- toc:true- branch: master- badges: true- comments: true
- author: Ijeoma Odoko
- categories: [python, power bi, jupyter]

## About

The purpose of this project is to create a dataframe capturing stock price data and calculating indicator values using Python, and then visualize it in Power BI in a simple dashboard. 

Key steps taken include: 
1. Create the virtual environment for Windows in Anaconda. 
2. Download the required libraries, and the stock price and volume data from yahoo finance using the pandas_datareader python API into a pandas dataframe. 
3. Calculate the stock indicator values using the TA-lib Python library. Use the Pandas Library if function not available in TA-lib.
4. Convert the dataframe(s) to csv files to use in Power BI.
5. Create dashboard with the indicator values in Power BI.

## Step 1: Create the Virtual Environment for Windows in Anaconda 

Create a new python 3.7 environment in Anaconda. Reference this [cheatsheet.](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)


Download the TA-Lib wrapper from [here.](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) To choose the right one you need to know the python version in your environment, and computer's operating system (32bit or 64bit).

Move it to the same Computer path that shows on your Anaconda Powershell prompt for your new virtual environment.

Run pip install with the TA-Lib file name, see picture below.

Pip Install all other required libraries like pandas, datetime, pandas_datareader.



![install%20datetime%20pandasdatareader.PNG](attachment:install%20datetime%20pandasdatareader.PNG)

## Step 2: Download required python libraries and data from Yahoo Finance


```python
# import python libraries
from pandas_datareader import data
import numpy as np
from datetime import datetime 
from datetime import date, timedelta
import pandas as pd
import talib
```


```python
# Create Parameters 
Number_of_days = input('Enter Calendar days:')
print(Number_of_days)

Stock_Ticker = input('Enter Stock Ticker:') 
print(Stock_Ticker)

Moving_Average = input('Enter Bollinger Bands Moving Average Type (Simple or Exponential): ')
if Moving_Average == 'Simple':
    print(Moving_Average)
elif Moving_Average == 'Exponential':
    print(Moving_Average)
else: 
    print('error')

Days = int(Number_of_days)

```

    Enter Calendar days:740
    740
    Enter Stock Ticker:TQQQ
    TQQQ
    Enter Bollinger Bands Moving Average Type (Simple or Exponential): Simple
    Simple
    


```python
# load stock data from yahoo 

start = date.today() - timedelta(Days)
today = date.today()
df = data.DataReader(Stock_Ticker, start=start, end=today,
                       data_source='yahoo')
```

## Step 3: Create stock indicator values using Pandas and TA-lib library


```python
# ADD INDICATORS TO MAIN DATAFRAME df

df['Daily_Returns'] = df['Adj Close'].pct_change()  # create column for daily returns
df['Price_Up_or_Down'] = np.where(df['Daily_Returns'] < 0, -1, 1) # create column for price up or down

## add columns for the volatility and volume indicators
from talib import ATR, OBV, ADX, RSI
df['Average_True_Range'] = ATR(df['High'], df['Low'], df['Close'])
df['On_Balance_Volume'] = OBV(df['Adj Close'], df['Volume'])

## add columns for momentum indicators
from talib import ADX, RSI, STOCH, WILLR, MFI
df['ADX'] = ADX(df['High'], df['Low'], df['Close'], timeperiod=14) #create column for ADX assume timeperiod of 14 days
df['RSI'] = RSI(df['Adj Close'],timeperiod=14) #create column for RSI assume timeperiod of 14 days 
df['William_%R'] = WILLR(df['High'],df['Low'], df['Close'], timeperiod=14) #create column for William %R use high, low and close, and assume timeperiod of 14 days
df['MFI'] = MFI(df['High'],df['Low'], df['Close'], df['Volume'], timeperiod=14) #create column for MFI use high, low and close, and assume timeperiod of 14 days

## add columns for statistic functions
from talib import LINEARREG, TSF

adj_close = df['Adj Close'].to_numpy()  # create ndarray from the adj_close prices

df['Linear_Regression'] = LINEARREG(adj_close, timeperiod=14)
df['Time_Series_Forecast'] = TSF(adj_close, timeperiod=14)

## add column for moving averages
from talib import MA, SMA, EMA, WMA    #import the moving average functions
df['Simple_Moving_Average_50'] = SMA(df['Adj Close'], timeperiod=50)
df['Simple_Moving_Average_200'] = SMA(df['Adj Close'], timeperiod=200)
    
## add columns for momentum indicators STOCH_df
slowk, slowd = STOCH(df['High'], df['Low'], df['Close'], fastk_period= 5, slowk_period= 3, slowk_matype= 0, slowd_period = 3, slowd_matype = 0) # uses high, low, close by default
STOCH_array = np.array([slowk, slowd]).transpose() 
STOCH_df = pd.DataFrame(data = STOCH_array, index=df.index, columns=['STOCH_slowk', 'STOCH_slowd'])

df = STOCH_df[['STOCH_slowk', 'STOCH_slowd']].join(df, how='right')  # join STOCH to main dataframe 



# CREATE Bollinger Bands® DATAFRAME WITH Bollinger Bands® INDICATORS

from talib import BBANDS

## Parameters
BBands_periods = 20
SDnumber = 2

if Moving_Average == 'Simple': 
    moving_avg= 0
    
else:
    moving_avg = 1    
 
upperband, middleband, lowerband = BBANDS(adj_close, timeperiod=BBands_periods, nbdevup=SDnumber, nbdevdn = SDnumber, matype=moving_avg)  # calculate the bollinger bands assuming the middle band as the simple moving average

bands = np.array([upperband, middleband, lowerband]).transpose()  # transpose the ndarrays 

stocks_bollingerbands = pd.DataFrame(data = bands, index = df.index, columns=['BB_upperband', 'BB_middleband', 'BB_lowerband']) # create dataframe from the ndarrays

stocks_bollingerbands = df[['Adj Close']].join(stocks_bollingerbands, how='left')  # add Adj Close Column and volume to bollinger bands dataframe 

stocks_bollingerbands['BB_Width'] = (stocks_bollingerbands['BB_upperband'] - stocks_bollingerbands['BB_lowerband']).div(stocks_bollingerbands['BB_middleband']) # add column for Bollinger Band Width

stocks_bollingerbands['Percent_B'] = (stocks_bollingerbands['Adj Close'] - stocks_bollingerbands['BB_lowerband']).div(stocks_bollingerbands['BB_upperband'] - stocks_bollingerbands['BB_lowerband']).mul(100) # add column for Percent B

## create list for Bollinger Bands® buy/sell strategy conditions
buy_or_sell = [
    (stocks_bollingerbands['Adj Close'] > stocks_bollingerbands['BB_upperband']),
    (stocks_bollingerbands['Adj Close'] <  stocks_bollingerbands['BB_lowerband'])
    ]

choices = ['Sell', 'Buy']

stocks_bollingerbands['Action'] = np.select(buy_or_sell, choices, default = 'Neutral') # create new column for action to take based on BollingerBand values

stocks_bollingerbands.drop(columns='Adj Close', inplace=True)  # drop Adj Close column

stocks_bollingerbands.dropna(inplace=True) # drop empty rows


# CREATE MERGED DATAFRAME

# join Bollinger Bands® dataframe to main dataframe to create new

stocks_df = stocks_bollingerbands.join(df, how='left')   
stocks_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BB_upperband</th>
      <th>BB_middleband</th>
      <th>BB_lowerband</th>
      <th>BB_Width</th>
      <th>Percent_B</th>
      <th>Action</th>
      <th>STOCH_slowk</th>
      <th>STOCH_slowd</th>
      <th>High</th>
      <th>Low</th>
      <th>...</th>
      <th>Average_True_Range</th>
      <th>On_Balance_Volume</th>
      <th>ADX</th>
      <th>RSI</th>
      <th>William_%R</th>
      <th>MFI</th>
      <th>Linear_Regression</th>
      <th>Time_Series_Forecast</th>
      <th>Simple_Moving_Average_50</th>
      <th>Simple_Moving_Average_200</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-10-30</th>
      <td>70.023204</td>
      <td>57.441679</td>
      <td>44.860155</td>
      <td>0.438063</td>
      <td>15.693175</td>
      <td>Neutral</td>
      <td>27.900666</td>
      <td>27.639096</td>
      <td>49.049999</td>
      <td>45.570000</td>
      <td>...</td>
      <td>5.123637</td>
      <td>-1.397404e+08</td>
      <td>NaN</td>
      <td>31.158793</td>
      <td>-71.739130</td>
      <td>32.891829</td>
      <td>49.628769</td>
      <td>48.881414</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-10-31</th>
      <td>67.553510</td>
      <td>56.486655</td>
      <td>45.419801</td>
      <td>0.391840</td>
      <td>30.191152</td>
      <td>Neutral</td>
      <td>46.609921</td>
      <td>35.213573</td>
      <td>53.660000</td>
      <td>51.250000</td>
      <td>...</td>
      <td>5.096948</td>
      <td>-1.172207e+08</td>
      <td>NaN</td>
      <td>36.433929</td>
      <td>-53.344484</td>
      <td>41.639084</td>
      <td>48.860358</td>
      <td>48.009042</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-11-01</th>
      <td>65.799976</td>
      <td>55.848475</td>
      <td>45.896975</td>
      <td>0.356375</td>
      <td>42.609367</td>
      <td>Neutral</td>
      <td>71.598957</td>
      <td>48.703181</td>
      <td>54.619999</td>
      <td>51.439999</td>
      <td>...</td>
      <td>4.960023</td>
      <td>-1.014624e+08</td>
      <td>NaN</td>
      <td>39.862677</td>
      <td>-40.635438</td>
      <td>40.177119</td>
      <td>49.358470</td>
      <td>48.622607</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-11-02</th>
      <td>64.439876</td>
      <td>55.214786</td>
      <td>45.989696</td>
      <td>0.334153</td>
      <td>32.264480</td>
      <td>Neutral</td>
      <td>83.134317</td>
      <td>67.114398</td>
      <td>54.930000</td>
      <td>50.799999</td>
      <td>...</td>
      <td>4.900736</td>
      <td>-1.232571e+08</td>
      <td>NaN</td>
      <td>37.529625</td>
      <td>-54.236345</td>
      <td>40.113450</td>
      <td>49.019171</td>
      <td>48.272671</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-11-05</th>
      <td>63.153036</td>
      <td>54.613530</td>
      <td>46.074023</td>
      <td>0.312725</td>
      <td>31.381121</td>
      <td>Neutral</td>
      <td>78.868933</td>
      <td>77.867402</td>
      <td>52.139999</td>
      <td>49.759998</td>
      <td>...</td>
      <td>4.720684</td>
      <td>-1.360822e+08</td>
      <td>NaN</td>
      <td>37.041639</td>
      <td>-57.079145</td>
      <td>34.302688</td>
      <td>49.342788</td>
      <td>48.745978</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-10-06</th>
      <td>138.348875</td>
      <td>125.399501</td>
      <td>112.450128</td>
      <td>0.206530</td>
      <td>54.172024</td>
      <td>Neutral</td>
      <td>31.246967</td>
      <td>50.686292</td>
      <td>135.119995</td>
      <td>125.150002</td>
      <td>...</td>
      <td>9.713780</td>
      <td>1.352882e+09</td>
      <td>20.178407</td>
      <td>48.428622</td>
      <td>-38.672007</td>
      <td>42.728684</td>
      <td>132.350001</td>
      <td>133.575605</td>
      <td>131.548601</td>
      <td>95.623291</td>
    </tr>
    <tr>
      <th>2020-10-07</th>
      <td>138.518796</td>
      <td>125.452501</td>
      <td>112.386207</td>
      <td>0.208307</td>
      <td>79.379039</td>
      <td>Neutral</td>
      <td>49.199414</td>
      <td>46.141732</td>
      <td>134.110001</td>
      <td>129.380005</td>
      <td>...</td>
      <td>9.564938</td>
      <td>1.381018e+09</td>
      <td>19.732322</td>
      <td>52.268168</td>
      <td>-16.483153</td>
      <td>49.831451</td>
      <td>134.138573</td>
      <td>135.507804</td>
      <td>132.055401</td>
      <td>95.863445</td>
    </tr>
    <tr>
      <th>2020-10-08</th>
      <td>139.695310</td>
      <td>125.985501</td>
      <td>112.275692</td>
      <td>0.217641</td>
      <td>83.204312</td>
      <td>Neutral</td>
      <td>55.772146</td>
      <td>45.406176</td>
      <td>136.610001</td>
      <td>133.539993</td>
      <td>...</td>
      <td>9.130299</td>
      <td>1.405755e+09</td>
      <td>18.965605</td>
      <td>53.370095</td>
      <td>-9.943310</td>
      <td>57.363264</td>
      <td>135.420572</td>
      <td>136.782528</td>
      <td>132.528001</td>
      <td>96.110250</td>
    </tr>
    <tr>
      <th>2020-10-09</th>
      <td>141.996039</td>
      <td>126.955001</td>
      <td>111.913964</td>
      <td>0.236951</td>
      <td>97.021374</td>
      <td>Neutral</td>
      <td>83.294339</td>
      <td>62.755300</td>
      <td>141.389999</td>
      <td>136.960007</td>
      <td>...</td>
      <td>8.928135</td>
      <td>1.430164e+09</td>
      <td>17.616138</td>
      <td>56.673091</td>
      <td>-0.900041</td>
      <td>64.342049</td>
      <td>137.913431</td>
      <td>139.403519</td>
      <td>133.084601</td>
      <td>96.386550</td>
    </tr>
    <tr>
      <th>2020-10-12</th>
      <td>147.458290</td>
      <td>128.269001</td>
      <td>109.079712</td>
      <td>0.299204</td>
      <td>117.722658</td>
      <td>Sell</td>
      <td>90.841647</td>
      <td>76.636044</td>
      <td>158.729996</td>
      <td>146.860001</td>
      <td>...</td>
      <td>9.549696</td>
      <td>1.466805e+09</td>
      <td>18.085055</td>
      <td>62.874384</td>
      <td>-9.019373</td>
      <td>64.782233</td>
      <td>144.046286</td>
      <td>146.150660</td>
      <td>133.782201</td>
      <td>96.717500</td>
    </tr>
  </tbody>
</table>
<p>491 rows × 26 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 510 entries, 2018-10-03 to 2020-10-12
    Data columns (total 20 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   STOCH_slowk                502 non-null    float64
     1   STOCH_slowd                502 non-null    float64
     2   High                       510 non-null    float64
     3   Low                        510 non-null    float64
     4   Open                       510 non-null    float64
     5   Close                      510 non-null    float64
     6   Volume                     510 non-null    float64
     7   Adj Close                  510 non-null    float64
     8   Daily_Returns              509 non-null    float64
     9   Price_Up_or_Down           510 non-null    int32  
     10  Average_True_Range         496 non-null    float64
     11  On_Balance_Volume          510 non-null    float64
     12  ADX                        483 non-null    float64
     13  RSI                        496 non-null    float64
     14  William_%R                 497 non-null    float64
     15  MFI                        496 non-null    float64
     16  Linear_Regression          497 non-null    float64
     17  Time_Series_Forecast       497 non-null    float64
     18  Simple_Moving_Average_50   461 non-null    float64
     19  Simple_Moving_Average_200  311 non-null    float64
    dtypes: float64(19), int32(1)
    memory usage: 101.7 KB
    


```python
stocks_bollingerbands.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 491 entries, 2018-10-30 to 2020-10-12
    Data columns (total 6 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   BB_upperband   491 non-null    float64
     1   BB_middleband  491 non-null    float64
     2   BB_lowerband   491 non-null    float64
     3   BB_Width       491 non-null    float64
     4   Percent_B      491 non-null    float64
     5   Action         491 non-null    object 
    dtypes: float64(5), object(1)
    memory usage: 46.9+ KB
    


```python
stocks_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 491 entries, 2018-10-30 to 2020-10-12
    Data columns (total 26 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   BB_upperband               491 non-null    float64
     1   BB_middleband              491 non-null    float64
     2   BB_lowerband               491 non-null    float64
     3   BB_Width                   491 non-null    float64
     4   Percent_B                  491 non-null    float64
     5   Action                     491 non-null    object 
     6   STOCH_slowk                491 non-null    float64
     7   STOCH_slowd                491 non-null    float64
     8   High                       491 non-null    float64
     9   Low                        491 non-null    float64
     10  Open                       491 non-null    float64
     11  Close                      491 non-null    float64
     12  Volume                     491 non-null    float64
     13  Adj Close                  491 non-null    float64
     14  Daily_Returns              491 non-null    float64
     15  Price_Up_or_Down           491 non-null    int32  
     16  Average_True_Range         491 non-null    float64
     17  On_Balance_Volume          491 non-null    float64
     18  ADX                        483 non-null    float64
     19  RSI                        491 non-null    float64
     20  William_%R                 491 non-null    float64
     21  MFI                        491 non-null    float64
     22  Linear_Regression          491 non-null    float64
     23  Time_Series_Forecast       491 non-null    float64
     24  Simple_Moving_Average_50   461 non-null    float64
     25  Simple_Moving_Average_200  311 non-null    float64
    dtypes: float64(24), int32(1), object(1)
    memory usage: 121.7+ KB
    

## Step 4: Download dataframes to csv



Power BI only supports the following python packages: [source](https://powerbi.microsoft.com/en-us/blog/python-visualizations-in-power-bi-service/)

  
  *matplotlib*
  
  *numpy*
  
  *pandas*
  
  *scikit-learn*
  
  *scipy*
  
  *seaborn*
  
  *statsmodels*


```python
## send to CSV file

df.to_csv(r"C:\TQQQ_df.csv", sep = ',')
stocks_bollingerbands.to_csv(r"C:\TQQQ_bbands.csv", sep=',')

```

## Step 5: Create visual dashboard in Power BI using the dataframes 

![Power%20BI%20-%20Stock%20Price%20Indicator%20Dashboard.PNG](attachment:Power%20BI%20-%20Stock%20Price%20Indicator%20Dashboard.PNG)

## References

[Fidelity, Percent B (% B).](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/percent-b#:~:text=How%20this%20indicator%20works,Percent%20B%20is%2050%20percent.) Accessed October 1, 2020.

[Fidelity, Bollinger Band® Width.](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-band-width) Accessed October 1, 2020.

[Fidelity, Bollinger Bands®.](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands) Accessed October 1, 2020.

[TA-lib python library.](https://github.com/mrjbq7/ta-lib) Accessed October 1, 2020.

[Williams %R.](https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r) Accessed October 9, 2020

[Money Flow Index](https://www.investopedia.com/terms/m/mfi.asp)Accessed October 9, 2020.

[Relative Strength Index](https://www.investopedia.com/terms/r/rsi.asp)Accessed October 9, 2020.

[ADX: The Trend Strength Indicator](https://www.investopedia.com/articles/trading/07/adx-trend-indicator.asp)Accessed October 9, 2020.

[Bollinger Bands Rules.](https://www.bollingerbands.com/bollinger-band-rules) Accessed October 1, 2020.




```python

```
