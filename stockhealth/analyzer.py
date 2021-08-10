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
        df['Increase [%]'] = df['Close'].diff(periods=1) / df['Open'] * 100
        df['Volatility'] = (df['High'] - df['Low']) / df['Open'] * 100
        df['Real Worth'] = df['Close'][0] + \
            (df['Close'].diff(periods=1) - df['Dividends']).cumsum()
        df['Real Worth'].values[0] = df['Close'].values[0]
        df['Increase in Real Worth [%]'] = df['Real Worth'].diff(periods=1) / \
            df['Real Worth'].shift(periods=1) * 100

    def trends(
            self,
            window: int = int(10),
            key_param: str = 'Close',
    ) -> tuple:
        """Get the technical analysis trends."""
        fig = plt.figure('Trends analysis: ' + self.__symbol)
        axs = fig.subplots(nrows=2, ncols=1, sharex=True)
        df = self.__history
        df[key_param].rolling(
            window=window, closed='right'
        ).mean().plot(
            style='-.', color='k', label='mean', ax=axs[0],
        )
        df[key_param].rolling(
            window=window, closed='right',
        ).max().plot(style='--', color='g', label='max', ax=axs[0],)
        df[key_param].rolling(
            window=window, closed='right',
        ).min().plot(style='--', color='r', label='min', ax=axs[0],)
        up_trend = df[df['Open'] < df['Close']]
        down_trend = df[df['Open'] > df['Close']]
        side_trend = df[df['Open'] == df['Close']]
        for trend_id, color_id in zip(
            [up_trend, down_trend, side_trend], ['g', 'r', 'b'],
        ):
            trend_id['Open'].plot(
                style='>', color=color_id, ax=axs[0],
            )
            trend_id['Close'].plot(
                style='<', color=color_id, ax=axs[0],
            )
            axs[0].vlines(
                x=trend_id.index,
                ymin=trend_id['Low'],
                ymax=trend_id['High'],
                color=color_id,
            )
        axs[0].legend(['mean', 'max', 'min'])
        axs[0].set_title('Value')
        axs[0].set_ylabel('Value [$]')
        df['Volume'].plot(
            ax=axs[1], style='-', color='k', grid=True, title='Volume',
        )
        axs[1].set_ylabel('Volume [-]')
        axs[1].set_yscale('log')
        _ = [ax.grid(True) for ax in axs]
        fig.suptitle('Trends analysis')
        return fig, axs
