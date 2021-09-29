#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:12:23 2020

Individual programming project: Stock Quotes Application

@author: Tianyi Zhang (19202673)

Based on the code from tutorials in MIS41110 Programming for Analytics of UCD, @D. Redmond
Based on the code in tutorial at this website: https://www.bilibili.com/video/BV1da4y147iW (descriptive part)
Based on the code in tutorial at this website: https://mp.weixin.qq.com/s/59FhX-puUUEHQjJKzIDjow （predictive part）


In descriptive analytics, the application can gather historical stock quotes according to customized requirements.
The techniques provided are statistical description, Candlestick chart (daily, weekly, monthly), Moving Average,
Exponentially Weighted Moving Average, Moving Average Convergence Divergence, and Scatter chart.

In predictive analytics, you are able to get the predicted closing price of the stock in the next few days.
Root-mean-square error and r^2 will be provided to judge the credibility of the prediction results.

"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ochl
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense

# max columns to output
pd.set_option('display.max_columns', 8)

# max rows to output
pd.set_option('display.max_rows', 2000)

# to output normal number rather than Scientific notation
np.set_printoptions(suppress=True)

# to output normal number rather than Scientific notation
pd.set_option('display.float_format', lambda x: '%.2f' % x)


def greeting():
    """
    say hello to users
    """
    print('Hi, Welcome to Tianyi\'s Stock Quotes Application!')
    print('=' * 50)


def function_select():
    """
    show choices of this application and for people to choose
    :return: string type, the number input by user
    """
    print('\nThere are 5 choices for you:')
    print('1. Descriptive Analytics')
    print('2. Predictive Analytics')
    print('3. Export Data')
    print('4. Instructions for use')
    print('5. Quit')

    # get a number from user
    number = input('Please enter a number: ')
    return number


def instructions_for_users():
    """
    Reading User Guide
    """
    guide = open('user_manual.txt', encoding='UTF-8')
    instructions = guide.read()
    print(instructions)
    guide.close()

    # Judge the user's choice
    next_step = input('\nDo you want to start analysing? (Y/N): ').upper()
    if next_step == 'Y':
        number = function_select()
        company = load_company_list()
        process_choice(number, company)
    else:
        quit()


def load_company_list():
    """
    get company list for searching Stocks
    :return: pandas DataFrame, a company list
    """
    return pd.read_csv('./companylist.csv')


def format_company_list(company_list):
    """
    searching stock's symbol and call function query_time_series
    :param company_list: pandas DataFrame
    :return val: pandas DataFrame
    :return clist: pandas DataFrame
    """
    print('')
    print('Search Stocks')
    print('=' * 50)

    # clist --> create a new DataFrame
    clist = company_list
    clist.sort_index()

    # delete the column which is all NaN
    clist.dropna(axis=1, how='all', inplace=True)
    clist['index'] = [i for i in range(clist.shape[0])]

    return clist


def get_symbol_name(clist):
    """
    only to show the 'Symbol' and 'Name' columns avoiding too much information on the output screen
    :param clist: pandas DataFrame
    :return: pandas DataFrame
    """
    return clist[['Symbol', 'Name']]


def search_symbol(company_symbol_name, c_list):
    """
    search for symbol according to the input of customers
    :param company_symbol_name: pandas DataFrame
            (index: Symbol, Name, LastSale, MarketCap, IPOyear, Sector, industry, Summary Quote")
    :param c_list: pandas DataFrame
            (index: Symbol and Name)
    :return: string
    """
    val = company_symbol_name
    clist = c_list
    symbol = input("Please input ticker symbol or company name: ").lower()
    filtered_companies = val[
        (clist.Symbol.str.lower().str.contains(symbol)) | (clist.Name.str.lower().str.contains(symbol))]

    # Determine if there is such a symbol or company name
    while len(filtered_companies.index) == 0:
        print('There\'s no such symbol or company, please try again!')
        search_symbol(val, clist)

    print(filtered_companies)
    symbol_chosen = input('\nPlease input the Symbol: ').upper()

    return symbol_chosen

    # # after searching，do "Query Time Range", call query_time_series to do descriptive and predictive analytics
    # query_time_series()


def get_all_historical_quotes(symbol_chosen):
    """
    According to the symbol entered by the user, Get all the historical stock quotes from the Internet.
    :param symbol_chosen: string
    :return: pandas DataFrame
    """
    symbol_choice = symbol_chosen
    # Get historical stock quotes on the Internet according to the symbol entered by the user
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey=QGO6Z4WQY7X2ZY1V&datatype=csv'.format(
        symbol_choice)
    data = pd.read_csv(url)

    # Process the format of historical quotes obtained
    data = data.sort_index()
    data['time'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='time', ascending=True)
    data['index'] = [i for i in range(data.shape[0])]
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')

    return data


def judgement_1(company, stock_data):
    """
    Judge the user's choice
    :param company: .csv file
    :param stock_data: pandas DataFrame
    """
    store = input('\nDo you want to export data? (Y/N): ').upper()
    if store == 'Y':
        export_data(stock_data)
    else:
        number = function_select()
        company = load_company_list()
        process_choice(number, company)


def get_start_day():
    """
    get the start date the user want to consult
    :return: Date
    """
    print('')
    print('Query Time Range')
    print('=' * 50)
    print('Please input a time range')
    return input('start time (yyyy-mm-dd): ')


def get_end_day():
    """
    get the end date the user want to consult
    :return: Date
    """
    return input('end time (yyyy-mm-dd): ')


def query_time_series(data, start_date, end_date):
    """
    According to the set date range, get the corresponding stock historical quotation
    :param data: pandas DataFrame
    :param start_date: Date (yyyy-mm-dd)
    :param end_date: Date (yyyy-mm-dd)
    :return: pandas DataFrame
    """
    all_historical_quotes = data
    # Obtain historical quotes for the corresponding time period according to the date entered by the user
    con1 = all_historical_quotes['time'] >= start_date
    con2 = all_historical_quotes['time'] <= end_date
    data_chosen = all_historical_quotes[con1 & con2]

    return data_chosen


def show_data(data, open_day, end_day, symbol_choice):
    """
    the overall describe of selected stock quotes: count, mean, std, 25%, 75%, median
    :param data: pandas DataFrame
    :param open_day: Date (yyyy-mm-dd)
    :param end_day: Date (yyyy-mm-dd)
    :param symbol_choice: string
    """
    data_chosen = data
    data_shown = data_chosen[['open', 'close', 'high', 'low', 'volume']]
    print('')
    print('The overall description of {}, from {} to {}'.format(symbol_choice, open_day, end_day))
    print('*' * 65)
    describe = data_shown.describe()

    # calculate Coefficient of Variation
    mean = describe.loc['mean']
    std = describe.loc['std']
    coefficient = mean / std

    # change coefficient to a pandas dataframe
    coefficient = coefficient.to_frame()
    coefficient = coefficient.T
    coefficient.index = ['Coefficient of Variation']

    # concat describe and coefficient
    overall_describe = pd.concat([describe, coefficient], axis=0)

    # Visualization
    print(overall_describe)


def get_k_type():
    """
    get the kind of stock candlestick chart the user  want to see
    :return: string: 1. daily 2. weekly 3. monthly
    """
    K_type = input(
        '\nWhat kind of stock candlestick chart do you want to see? \n(1. daily 2. weekly 3. monthly): ')
    return K_type


def get_MA_period():
    """
    get the time range for Moving Average
    :return: int type, a number (time range (unit: day)) which will pass to function moving_average
    """
    SAM_period = int(
        input('\nPlease enter the time range (unit: day) \nyou expect for Moving Average (enter a number): '))
    return SAM_period


def k_daily(data, start_day, end_day, symbol_choice):
    """
    plotting Daily Candlestick chart
    :param data: pandas DataFrame
    :param start_day: Date (yyyy-mm-dd)
    :param end_day: Date (yyyy-mm-dd)
    :param symbol_choice: string
    """
    data_chosen = data
    valu_day = data_chosen[['index', 'open', 'close', 'high', 'low']].values
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 8), dpi=80)
    candlestick_ochl(axes, valu_day, width=0.2, colorup='r', colordown='g')

    # Visualization
    plt.xlabel('Date Range from {} to {}'.format(start_day, end_day))
    plt.title('Daily Candlestick for {}'.format(symbol_choice))
    plt.show()


def k_weekly(data, start_day, end_day, symbol_choice):
    """
    plotting weekly Candlestick chart
    :param data: pandas DataFrame
    :param start_day: Date (yyyy-mm-dd)
    :param end_day: Date (yyyy-mm-dd)
    :param symbol_choice: string
    """
    data_chosen = data

    # resample as  a stock for each week
    stock_week_k = data_chosen.resample('w').last()
    stock_week_k['open'] = data_chosen['open'].resample('w').first()
    stock_week_k['close'] = data_chosen['close'].resample('w').last()
    stock_week_k['high'] = data_chosen['high'].resample('w').max()
    stock_week_k['low'] = data_chosen['low'].resample('w').min()
    stock_week_k['volume'] = data_chosen['volume'].resample('w').sum()

    # reset index
    stock_week_k['index'] = [i for i in range(stock_week_k.shape[0])]

    # Visualization
    valu_week = stock_week_k[['index', 'open', 'close', 'high', 'low']].values
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 8), dpi=80)
    candlestick_ochl(axes, valu_week, width=0.2, colorup='r', colordown='g')
    plt.xlabel('Date Range from {} to {}'.format(start_day, end_day))
    plt.title('Weekly Candlestick for {}'.format(symbol_choice))
    plt.show()


def k_monthly(data, start_day, end_day, symbol_choice):
    """
    plotting monthly Candlestick chart
    :param data: pandas DataFrame
    :param start_day: Date (yyyy-mm-dd)
    :param end_day: Date (yyyy-mm-dd)
    :param symbol_choice: string
    """
    data_chosen = data

    # resample as  a stock for each month
    stock_month_k = data_chosen.resample('m').last()
    stock_month_k['open'] = data_chosen['open'].resample('m').first()
    stock_month_k['close'] = data_chosen['close'].resample('m').last()
    stock_month_k['high'] = data_chosen['high'].resample('m').max()
    stock_month_k['low'] = data_chosen['low'].resample('m').min()
    stock_month_k['volume'] = data_chosen['volume'].resample('m').sum()

    # reset index
    stock_month_k['index'] = [i for i in range(stock_month_k.shape[0])]
    valu_month = stock_month_k[['index', 'open', 'close', 'high', 'low']].values

    # Visualization
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 8), dpi=80)
    candlestick_ochl(axes, valu_month, width=0.6, colorup='r', colordown='g')
    plt.xlabel('Date Range from {} to {}'.format(start_day, end_day))
    plt.title('Monthly Candlestick for {}'.format(symbol_choice))
    plt.show()


def moving_average(data, SAM_period, symbol_choice):
    """
    plotting moving average and Exponentially Weighted Moving Average chart
    :param data: pandas DataFrame
    :param SAM_period: int
    :param symbol_choice: string
    :return:
    """
    data_chosen = data

    # Simple Moving Average
    pd.Series.rolling(data_chosen['close'], window=SAM_period).mean().plot(figsize=(20, 8),
                                                                           label='Simple Moving Average')

    # Exponentially Weighted Moving Average
    pd.Series.ewm(data_chosen['close'], span=SAM_period).mean().plot(figsize=(20, 8),
                                                                     label='Exponentially Weighted Moving '
                                                                           'Average')

    # Visualization
    plt.legend(loc='best')
    plt.title('{}'.format(symbol_choice))
    plt.show()


def MACD(data, start_day, end_day, symbol_choice):
    """
    plotting Moving Average Convergence Divergence chart
    :param data: pandas DataFrame
    :param start_day: Date (yyyy-mm-dd)
    :param end_day: Date (yyyy-mm-dd)
    :param symbol_choice: string
    """
    data_chosen = data
    # organize values of data_chosen
    val = data_chosen[['index', 'open', 'close', 'high', 'low']]
    val['index'] = [i for i in range(val.shape[0])]

    # plotting
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 8), dpi=80)
    candlestick_ochl(axes, val.values, width=0.2, colorup='r', colordown='g')

    # MACD-->dif, macdsignal-->dea, macdhist-->macdbar
    dif, dea, macdbar = talib.MACD(val['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)

    # x-aix
    x = [i for i in range(val.shape[0])]

    # If it is a positive value, the output is a red bar. If it is negative, the output is a green bar
    bar1 = np.where(macdbar > 0, macdbar, 0)
    bar2 = np.where(macdbar < 0, macdbar, 0)
    plt.bar(x, bar1, color='r', label='up')
    plt.bar(x, bar2, color='g', label='down')

    # Visualization
    plt.plot(x, dif, label='MACD')
    plt.plot(x, dea, label='MACD_signal')
    plt.xlabel('Date Range from {} to {}'.format(start_day, end_day))
    plt.title('Moving Average Convergence Divergence for {}'.format(symbol_choice))
    plt.legend(loc='best')
    plt.show()


def scatter(data, symbol_choice):
    """
    plotting Scatter plots for each pair of variables
    :param data: pandas DataFrame
    :param symbol_choice: string
    :return:
    """
    # organize values of data_chosen
    data_chosen = data

    scatter_data = data_chosen.sort_index()
    scatter_data['index'] = [i for i in range(scatter_data.shape[0])]
    frame = scatter_data[['open', 'close', 'high', 'low', 'volume']]

    # Visualization
    pd.plotting.scatter_matrix(frame, figsize=(15, 15))
    plt.title('Scatter plot of each indicator pairwise correlation for {}'.format(symbol_choice))
    plt.legend(loc='best')
    plt.show()


def k_chart(K_type, data, start_day, end_day, symbol_choice):
    """
    Get the return value in function get_k_type, and determine the type of Stock Candlestick chart to plot
    :param K_type: string
    :param data: pandas DataFrame
    :param start_day: Date(yyyy-mm-dd)
    :param end_day: Date (yyyy-mm-dd)
    :param symbol_choice: string
    """
    if K_type == '1':
        k_daily(data, start_day, end_day, symbol_choice)
    elif K_type == '2':
        k_weekly(data, start_day, end_day, symbol_choice)
    elif K_type == '3':
        k_monthly(data, start_day, end_day, symbol_choice)
    else:
        print('Invalid choice! Try again please.')
        print('')
        k_chart(K_type, data, start_day, end_day, symbol_choice)


def judgement_2():
    """
    Judge the user's choice
    """
    next_step = input('\nDo you want to analyse the stock of another company? (Y/N): ').upper()
    if next_step == 'Y':
        number = function_select()
        company = load_company_list()
        process_choice(number, company)
    else:
        quit()


def get_size(data):
    """
    get the total number of historical stock quotes
    :param data: pandas DataFrame
    :return: int
    """
    data_predict = data
    return int(data_predict.shape[0] * 0.8)


def get_predict_data(data):
    """
    sort the obtained pandas DataFrame
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    all_quotes = data
    data_predict = all_quotes[['open', 'close', 'high', 'low', 'volume']]
    data_predict = data_predict.sort_index(axis=0, ascending=True)
    return data_predict


def get_days_to_predict():
    """
    Users enter how many days they want to predict backwards
    :return: int
    """
    return int(input('Please input how many days do you want to predict (a number): '))


def get_time_stamp():
    """
    based on how many days do the user prefer to predict the next closing price.
    :return: int
    """
    return int(
        input('Based on how many days do you want to predict the next closing price? (enter a number): '))


def get_train_data_set(data, data_size, time_stamp):
    """
    Divide the acquired data into training set and validation set, 80% of the historical stock quotes
    is used as the training set
    :param data: pandas DataFrame
    :param data_size: int
    :param time_stamp: int
    :return: DataFrame
    """
    data_predict = data
    return data_predict[0:data_size + time_stamp]


def get_valid_data_set(data, data_size, time_stamp):
    """
    Divide the acquired data into training set and validation set, 20% of the historical stock quotes
    is used as the validation set
    :param data: pandas DataFrame
    :param data_size: int
    :param time_stamp: int
    :return: pandas DataFrame
    """
    data_predict = data
    return data_predict[data_size - time_stamp:]


def get_scaler():
    """
    Normalization parameter
    :return: MinMaxScaler()
    """
    return MinMaxScaler(feature_range=(0, 1))


def get_scaled_training_data(scaler, train_data_set):
    """
    Normalize the training set
    :param scaler: MinMaxScaler()
    :param train_data_set: pandas DataFrame
    :return: pandas DataFrame
    """
    return scaler.fit_transform(train_data_set)


def get_scaled_validation_data(scaler, valid_data_set):
    """
    Normalize the validation set
    :param scaler: MinMaxScaler()
    :param valid_data_set:  pandas DataFrame
    :return: pandas DataFrame
    """
    return scaler.fit_transform(valid_data_set)


def train_data_x(scaled_data, time_stamp, train_data):
    """
    get scaled training data, and create to a numpy array
    :param scaled_data: pandas DataFrame
    :param time_stamp: int
    :param train_data: pandas DataFrame
    :return: numpy array
    """
    # get_scaled_training_data
    train = train_data
    x_train = []
    for i in range(time_stamp, len(train)):
        x_train.append(scaled_data[i - time_stamp:i])
    x_train = np.array(x_train)
    return x_train


def train_data_y(scaled_data, time_stamp, train_data):
    """
    get scaled training data, and create to a numpy array
    :param scaled_data: pandas DataFrame
    :param time_stamp: int
    :param train_data: pandas DataFrame
    :return: numpy array
    """
    y_train = []
    for i in range(time_stamp, len(train_data)):
        y_train.append(scaled_data[i, 3])
    y_train = np.array(y_train)
    return y_train


def valid_data_x(scaled_data, time_stamp, valid_data):
    """
    get scaled validation data, and create to a numpy array
    :param scaled_data: pandas DataFrame
    :param time_stamp: int
    :param valid_data: pandas DataFrame
    :return: numpy array
    """
    x_valid = []
    for i in range(time_stamp, len(valid_data)):
        x_valid.append(scaled_data[i - time_stamp:i])
    x_valid = np.array(x_valid)
    return x_valid


def valid_data_y(scaled_data, time_stamp, valid_data):
    """
    get scaled validation data, and create a numpy array
    :param scaled_data: pandas DataFrame
    :param time_stamp: int
    :param valid_data: pandas DataFrame
    :return: numpy array
    """
    y_valid = []
    for i in range(time_stamp, len(valid_data)):
        y_valid.append(scaled_data[i, 3])
    y_valid = np.array(y_valid)
    return y_valid


def LSTM_model(scaler, x_train, y_train, valid_data, x_valid):
    """
    predicting the closing price by using Long short-term memory (an artificial recurrent neural network)
    :param scaler: MinMaxScaler()
    :param x_train: numpy array
    :param y_train: numpy array
    :param valid_data: pandas DataFrame
    :param x_valid: numpy array
    :return: numpy array: the predicting closing price
    """
    # Hyper-parameter settings
    epochs = 3
    batch_size = 16
    # LSTM Parameters: return_sequences=True.
    # The LSTM output is a sequence. The default is False, and a value is output.
    # input_dim：Enter the dimension of a single sample feature
    # input_length：the length of time entered
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_dim=x_train.shape[-1], input_length=x_train.shape[1]))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Forecast stock closing price
    closing_price = model.predict(x_valid)
    scaler.fit_transform(pd.DataFrame(valid_data['close'].values))

    # Denormalize
    closing_price = scaler.inverse_transform(closing_price)

    return closing_price


def denormalize_valid_y(scaler, y_valid):
    """
    denormalize the normalized y_valid
    :param scaler: MinMaxScaler()
    :param y_valid: numpy array
    :return: numpy array
    """
    return scaler.inverse_transform([y_valid])


def root_mean_squared(y_valid, closing_price):
    """
    root mean squared of predicted pricing
    :param y_valid: numpy array
    :param closing_price: numpy array
    """
    print('\nRoot Mean Squared Error and R^2: ')
    RMSE = np.sqrt(np.mean(np.power((y_valid - closing_price), 2)))
    print('RMSE = {}'.format(RMSE))


def r2(y_valid, closing_price):
    """
    r^2 of predicted pricing
    :param y_valid: numpy array
    :param closing_price: numpy array
    """
    r2 = 1 - (np.sum(np.power((y_valid - closing_price), 2))) / (
        np.sum(np.power((y_valid - np.mean(closing_price)), 2)))
    print('r^2 = {}'.format(r2))


def format(y_valid, closing_price):
    """
    combine y_valid and closing_price, and then transfer to pandas DataFrame
    :param y_valid: numpy array
    :param closing_price: numpy array
    :return: pandas DataFrame
    """
    dict_data = {
        'predictions': closing_price.reshape(1, -1)[0],
        'close': y_valid[0]
    }
    data_pd = pd.DataFrame(dict_data)
    return data_pd


def output_predict_close_data(data, days, symbol_choice):
    """
    :param data: pandas DataFrame
    :param days: int
    :param symbol_choice: string
    """
    days_to_predict = days
    data_output = data[['predictions']].tail(days_to_predict)
    data_output.sort_index()
    data_output['index'] = [j for j in range(data_output.shape[0])]
    data_output = data_output.set_index('index')
    print('\nHere are the predictions of {}\n'.format(symbol_choice))
    print(data_output)


def curve_fitting(data):
    """
    Visualization and Curve fitting
    :param data: pandas DataFrame
    """
    plt.figure(figsize=(16, 8))
    plt.plot(data[['close']], label='closing price')
    plt.plot(data[['predictions']], label='predictions')
    plt.title('Long Short-Term Memory')
    plt.legend(loc='best')
    plt.show()


def quit():
    sure = input('\nAre you sure to quit? (Y/N):').upper()
    if sure == 'Y':
        print('Thanks for using! Have a good time.')
        sys.exit(0)
    else:
        number = function_select()
        company = load_company_list()
        process_choice(number, company)


def descriptive_analytics(company):
    """
    :param company: .csv file
    get basic information of user preferences
    to show overall description of stocks for chosen_date range
    Output stock candlestick chart, MA chart, MACD chart and Scatter plots for each pair of variables respectively.
    """
    print('\nNow, you are going to the descriptive analytics!')
    # get basic information of user preferences
    c_list = format_company_list(company)
    c_symbol_name = get_symbol_name(c_list)
    symbol = search_symbol(c_symbol_name, c_list)
    historical_quotes = get_all_historical_quotes(symbol)
    start_date = get_start_day()
    end_date = get_end_day()

    historical_quotes_selected = query_time_series(historical_quotes, start_date, end_date)
    k_type = get_k_type()
    period = get_MA_period()

    # to show all the graphs to the user
    show_data(historical_quotes_selected, start_date, end_date, symbol)
    k_chart(k_type, historical_quotes_selected, start_date, end_date, symbol)
    moving_average(historical_quotes_selected, period, symbol)
    MACD(historical_quotes_selected, start_date, end_date, symbol)
    scatter(historical_quotes_selected, symbol)

    judgement_1(company, historical_quotes_selected)


def predictive_analytics(company):
    """
    :param company: .csv file
    predicting the closing price by using Long short-term memory (an artificial recurrent neural network)
    """
    print('\nNow, you are going to the predictive analytics!')

    # get basic information of user preferences
    c_list = format_company_list(company)
    c_symbol_name = get_symbol_name(c_list)
    symbol = search_symbol(c_symbol_name, c_list)
    historical_quotes = get_all_historical_quotes(symbol)

    # prepare works for predicting
    size = get_size(historical_quotes)
    data_for_predict = get_predict_data(historical_quotes)
    days = get_days_to_predict()
    time_stamp = get_time_stamp()
    train = get_train_data_set(data_for_predict, size, time_stamp)
    valid = get_valid_data_set(data_for_predict, size, time_stamp)

    # normalization
    scaler = get_scaler()
    scaled_train_data = get_scaled_training_data(scaler, train)
    scaled_valid_data = get_scaled_validation_data(scaler, valid)
    x_train = train_data_x(scaled_train_data, time_stamp, train)
    y_train = train_data_y(scaled_train_data, time_stamp, train)
    x_valid = valid_data_x(scaled_valid_data, time_stamp, valid)
    y_valid = valid_data_y(scaled_valid_data, time_stamp, valid)

    # predicting results
    closing_price = LSTM_model(scaler, x_train, y_train, valid, x_valid)
    y_valid = denormalize_valid_y(scaler, y_valid)

    # Root Mean Squared Error and R^2
    root_mean_squared(y_valid, closing_price)
    r2(y_valid, closing_price)

    # Visualization and Curve fitting
    data_pd = format(y_valid, closing_price)
    output_predict_close_data(data_pd, days, symbol)
    curve_fitting(data_pd)
    judgement_2()


def export_data(stock_data):
    """
    Output the value of Simple Moving Average and Exponentially Weighted Moving Average with a period of 5/30/60/120
    days, and save them as a local .csv file
    :param stock_data: pandas DataFrame
    """
    print('')
    print('Export Data')
    print('=' * 50)

    day_list = [5, 30, 60, 120]
    # Simple Moving Average
    for i in day_list:
        stock_data['SMA' + str(i)] = pd.Series.rolling(stock_data['close'], window=i).mean()

    # Exponentially Weighted Moving Average
    for j in day_list:
        stock_data['EWMA' + str(j)] = pd.Series.ewm(stock_data['close'], span=j).mean()

    # output the analysed data to the user's computer and save as a .csv file called HistoricalQuotesWithAnalysis
    stock_data.to_csv('./HistoricalQuotesWithAnalysis.csv')

    print('\nData exported successfully!')

    # MAIN LOOP
    number = function_select()
    company = load_company_list()
    process_choice(number, company)


def process_choice(number, company_list):
    """
    get the inputted number of users and load company_list for descriptive and predictive analysing
    :param number: String
    :param company_list: pandas DataFrame
    """
    while number != '1' and number != '2' and number != '3' and number != '4' and number != '5':
        print('Sorry, invalid choice, please try again!')
        number = function_select()
        company = load_company_list()
        process_choice(number, company)

    if number == '1':
        descriptive_analytics(company_list)
    elif number == '2':
        predictive_analytics(company_list)
    elif number == '3':
        print('\nThere is no data for you to export!')
        print('You should do descriptive analytics first!')
        number = function_select()
        company = load_company_list()
        process_choice(number, company)
    elif number == '4':
        instructions_for_users()
    elif number == '5':
        quit()


def main():
    """
    main function
    """
    greeting()
    number = function_select()
    company = load_company_list()
    process_choice(number, company)


if __name__ == '__main__':
    main()
