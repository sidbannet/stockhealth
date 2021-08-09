#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth market data analyzer
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

import matplotlib.pyplot as plt
import yfinance as yf


class TimeSeries:
    """Time series analysis of a ticker symbol."""

    def __init__(
            self,
            ticker: str = '',
    ):
        """Instantiate the class."""
        self.__ticker = yf.Ticker(ticker)
        self.__symbol = ticker
        self.__history = self.__ticker.history(period='max', interval='1d')
        df = self.__history
        df['Increase'] = df['Close'].diff(periods=1) / df['Open'] * 100
        df['Volatility'] = (df['High'] - df['Low']) / df['Open'] * 100
        df['Real Worth'] = df['Close'][0] + \
            (df['Close'].diff(periods=1) - df['Dividends']).cumsum()
        df['Increase in Real Worth'] = df['Real Worth'].diff(periods=1) / \
            df['Real Worth'].shift(periods=1) * 100

    def trends(
            self,
            window: int = int(10),
    ) -> tuple:
        """Get the technical analysis trends."""
        fig = plt.figure('Trends analysis: ' + self.__symbol)
        axs = fig.subplots(nrows=1, ncols=1)
        df = self.__history
        df['Real Worth'].rolling(
            window=window, closed='right'
        ).mean().plot(
            style='-.', color='k', label='mean', ax=axs,
        )
        df['Real Worth'].rolling(
            window=window, closed='right',
        ).max().plot(style='--', color='r', label='max', ax=axs,)
        df['Real Worth'].rolling(
            window=window, closed='right',
        ).min().plot(style='--', color='b', label='min', ax=axs,)
        df['Real Worth'].plot(
            style='.', color='k', label='Worth', ax=axs,
            title='History', grid=True,
        )
        axs.legend()
        axs.set_ylabel('Real Worth [USD]')
        fig.suptitle('Trends analysis')
        return fig, axs
