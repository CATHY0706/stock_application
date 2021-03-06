# stock_application User Manual



## 1.	Using the Stock Quotes Application

### 1.1	About the Stock Quotes Application
The Stock Quotes Application is a computer program for consulting, analyzing, and modeling stock quotes data.
You can search for specific stock historical quotes and set designated time ranges.
This application can provide descriptive analytics and predictive analytics.
In descriptive analytics, the application can gather historical stock quotes according to your customized requirements.
The techniques provided are statistical description, Candlestick chart (daily, weekly, monthly), Moving Average,
Exponentially Weighted Moving Average, Moving Average Convergence Divergence, and Scatter chart.

In predictive analytics, you are able to get the predicted closing price of the stock in the next few days.
Root-mean-square error and coefficient of determination will be provided to you to judge the credibility of the prediction results.
This user guide outlines the features of the application and provides step-by-step instructions for completing various tasks.

### 1.2	Exploring the start function
* The start function provides a hub to different parts of the Stock Quotes application.
* Choose your preferred function by inputting the relevant choice.
* To do descriptive analytics, input 1 (see Chapter 2).
* To do predictive Analytics, input 2 (see Chapter 3).
* To export stock data, input 3 (see Chapter 4).
* To read instructions for use, input 4 (see Chapter 5).
* To quit, input 5 (see Chapter 6).

### 1.3	Getting additional help
For technical support and software assistance, please contact zhangtianyi630@gmail.com

## 2.	Descriptive Analytics
You can do as many descriptive analytics as you like and save the analysis results to your computer for reference or revision.

### 2.1	To search a specific stock
First, you can enter any key characters to search for a stock symbol.
For example, enter "zy", then the application will provide a list of stock symbols matching "zy".
According to the list provided by the application, input stock code you prefer.

### 2.2	To set designated time ranges
After determining the stock symbol, the application will ask you to input the time range (start date and end date).
Note: Use ???-??? to separate the year, month, and day

### 2.3	To choose the type of Candlestick chart
* There are three kinds of candlestick chart that the application can provide you.
* You need to choose one of them and input your choice.
* To get daily candlestick chart, input 1.
* To get weekly candlestick chart, input 2.
* To get monthly candlestick chart, input 3.
* In the above example, 2 is inputted. Thus, the application will show the weekly candlestick chart, see 2.5.1.

### 2.4 To set a time range for Moving Average
For moving average and exponentially weighted moving average techniques, you need to provide the time period for averaging.

### 2.5 To get descriptive statistics and graphs
#### 2.5.1 Statistical Analysis
* The statistical description of your selected stock symbol and the specific date range.
* Count: Total number of working dates in the selected period.
* Coefficient of Variation: std/mean.

#### 2.5.2 Candlestick chart

#### 2.5.3 Moving Average and Exponentially Weighted Moving Average

#### 2.5.4 Moving Average Convergence Divergence
#### Note: If you want to get a good MACD chart, it is recommended that you do not set a very small date range (ideally more than 10 months).

#### 2.5.5 Scatter
#### Note: The scatter plot shows the relationship between every two variables.

### 2.6 Export Data
After completing the descriptive analytics, you will be asked whether to export the data.
You need to provide the answer of ???Y/N???.

* To export, input Y. Not to export, input N.
* If your answer is ???N???, the application will automatically return to the start function.
* If your answer is ???Y???, see Chapter 4.



## 3.	Predictive Analytics
In the predictive analysis part, the method of long short-term memory is used.
Long short-term memory is a type of time recurrent neural network (RNN) architecture (Hochreiter and Schmidhuber, 1997).
As a nonlinear model, LSTM can be used to construct large-scale deep neural networks.

### 3.1 To search a specific stock
See 2.1

### 3.2 To input the number of days to forecast
Here, you are asked to input the number of days you want to predict.
See the following example, it indicates that the user wants to predict the closing stock price five days from today.


### 3.3 To set data training period
Here, you are asked to set a specific data training period.
The following example indicates that the next closing price will be predicted based on the closing price of the past 50 days.

* Note: Generally, in order to ensure the quality of forecast results, the prediction should be based on not less than 50 days of historical data.

### 3.4	 Curve fitting, Root mean squared error, and coefficient of determination
After finishing setting training period, the application will provide you with the curve fitting of predicted price and real closing price, root mean squared error, and coefficient of determination. These are the factors by which you determine whether the predicted closing price is credible.


### 3.5	To get predicted price
In 3.1, we assumed to predict the closing price of the next five days, so here, the application provides you with five closing prices.

After finishing predictive analytics, you will be asked that whether you want to continue analyzing.
To continue, input Y. Not to continue, input N.

If your answer is ???Y???, the application will automatically return to the start function. If your answer is ???N???, see Chapter 6.



## 4.	Export Data
Note: If you want to export data, you need to do descriptive analytics first.
Because the application do not have any stock quotes data without description analytics.

If you successfully completed the descriptive analytics, you would receive the following questions.

If you answer ???Y???, you will receive a reminder saying ???data exported successfully???,
then the application will automatically return to the start function.

A csv format file named ???HistoricalQuotesWithAnalysis??? will be exported to your computer.

The raw stock data is exported.
Meanwhile, the calculation result of the moving average and exponentially weighted moving average is also exported.



## 5.	Instructions for use
Users can also use this function to read the user guide of the application.
After finishing reading, the application will ask you if you want to start stock analytics.

To start, input Y. Not to start, input N.

If your answer is ???Y???, the application will automatically return to the start function.
If your answer is ???N???, see Chapter 6.


## 6.	Quit
If you choose to quit at any stage, then you need to confirm whether you really want to quit.

To quit, input Y. Not to quit, input N.

If your answer is ???N???, the application will automatically return to the start function again.
If your answer is ???Y???, then printing ???Thanks for using! Have a good time.??? And the whole procedure ends.

