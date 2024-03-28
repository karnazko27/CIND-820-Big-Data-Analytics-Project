#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import yfinance as yf
import pandas as pd
import numpy as np
# python standard datetime module
import datetime

def get_price_data(ticker, period="1d",
                         interval="1m"):
    """
    Gets price data
    """
    # create Ticker object
    ticker_obj = yf.Ticker(ticker)
    # get daily price data from Ticker object
    daily_price_data = ticker_obj.history(period=period,
                                    interval=interval)
    # convert datetime index to UTC timezone
    daily_price_data = daily_price_data.tz_convert('UTC')
    # reset index of DataFrame
    daily_price_data = daily_price_data.reset_index()
    daily_price_data["epoch_time"] = daily_price_data["Datetime"].apply(
        lambda row: pd.to_datetime(row).value // 1000000000
    )
    datetime_keys = daily_price_data["epoch_time"].values.astype(str)
    open_prices = np.round(daily_price_data["Open"].values, 2)
    high_prices = np.round(daily_price_data["High"].values, 2)
    low_prices = np.round(daily_price_data["Low"].values, 2)
    close_prices = np.round(daily_price_data["Close"].values, 2)
    new_dict = dict()

    for timestamp, open_price, close_price, high_price, low_price in zip (datetime_keys, open_prices, close_prices, high_prices, low_prices):
        new_dict[timestamp] = {"open": open_price,
                              "close": close_price,
                              "high": high_price,
                              "low": low_price}

    return new_dict
