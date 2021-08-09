#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth market data analyzer
# https://github.com/sidbannet
#
# Copyright 2021 Siddhartha Banerjee
#

import yfinance as yf


class TimeSeries:
    """Time series analysis of a ticker symbol."""

    def __init__(
            self,
            ticker: str = '',
    ):
        """Instantiate the class."""
        self.__ticker = yf.Ticker(ticker)
        self.__history = self.__ticker.history(period='max', interval='1d')
        df = self.__history
        df['Increase'] = df['Close'].diff(periods=1) / df['Open'] * 100
        df['Volatility'] = (df['High'] - df['Low']) / df['Open'] * 100
        df['Real Worth'] = df['Close'][0] + \
            (df['Close'].diff(periods=1) - df['Dividends']).cumsum()
        df['Increase in Real Worth'] = df['Real Worth'].diff(periods=1) / \
            df['Real Worth'].shift(periods=1) * 100
