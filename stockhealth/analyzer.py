#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth market data analyzer
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum, unique
import yfinance as yf
from stockhealth.simulation import MonteCarlo


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
        df['Intra-day volatility'] = (
            df['High'] - df['Low']
        ) / (
            df['Open']
        )
        df['Real Worth'] = df['Close'][0] + \
            (df['Close'].diff(periods=1) + df['Dividends']).cumsum()
        df['Real Worth'].values[0] = df['Close'].values[0]
        df['Risk free return'] = df['Real Worth'].diff(periods=1) / \
            df['Real Worth'].shift(periods=1)
        df['Risk free return'].values[0] = float(0.0)
        df['Volatility'] = (
            df['High'] - df['Low']
        ) / df['Real Worth'].shift(periods=1)
        df['Volatility'].values[0] = (
            df['High'].values[0] - df['Low'].values[0]
        ) / df['Open'].values[0]

    def technical(
            self,
            window: int = int(14),
            key_param: str = 'Close',
            volume_scale: str = 'log',
            ema: bool = True,
            adjust: bool = False,
            short_term: int = int(9),
            mid_term: int = int(12),
            long_term: int = int(26),
    ) -> tuple:
        """Get the technical analysis."""
        fig = plt.figure('Technical analysis: ' + self.__symbol)
        axs = fig.subplots(nrows=4, ncols=1, sharex=True)
        df = self.__history
        df[key_param].rolling(
            window=window, closed='right'
        ).mean().plot(
            style='-.', color='k', label='mean', ax=axs[0],
        )
        df[key_param].rolling(
            window=window, closed='right',
        ).max().plot(style='--', color='g', label='max', ax=axs[0], )
        df[key_param].rolling(
            window=window, closed='right',
        ).min().plot(style='--', color='r', label='min', ax=axs[0], )
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
        axs[0].legend(['Means', 'Highs', 'Lows'])
        axs[0].set_title('Charts')
        axs[0].set_ylabel('Value [$]')
        df['Volume'].plot(
            ax=axs[1], style='-', color='k', grid=True, title='Size',
        )
        axs[1].set_ylabel('Volume [-]')
        axs[1].set_yscale(volume_scale)
        (df['Close'] * 0 + 70).plot(
            style='--', color='r', ax=axs[2], label='SELL',
        )
        (df['Close'] * 0 + 30).plot(
            style='--', color='b', ax=axs[2], label='BUY',
        )
        self.rsi(periods=window, ema=ema).plot(
            style='-', color='k', ax=axs[2], label='RSI',
            title='Over trading indicator', grid=True,
        )
        axs[2].set_ylabel('RSI')
        axs[2].legend()
        macd = self.macd(
            long_span=long_term, mid_span=mid_term, short_span=short_term,
            adjust=adjust,
        )
        axs[3].fill_between(
            x=macd.index,
            y1=macd['MACD'],
            y2=macd['Signal'],
            where=(macd['MACD'] > macd['Signal']),
            color='g', alpha=0.7, interpolate=True,
        )
        axs[3].fill_between(
            x=macd.index,
            y1=macd['MACD'],
            y2=macd['Signal'],
            where=(macd['MACD'] < macd['Signal']),
            color='r', alpha=0.7, interpolate=True,
        )
        axs[3].set_title('Long term vs. Short term trends')
        axs[3].set_ylabel('MACD')
        _ = [ax.grid(True) for ax in axs]
        fig.suptitle('Technical analysis')
        return fig, axs

    def rsi(
            self,
            periods: int = int(14),
            ema: bool = True,
    ) -> pd.Series:
        """
        Returns a pd.Series with the relative strength index.
        """
        df = self.__history
        close_delta = df['Close'].diff()
        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
        if ema:
            # Use exponential moving average
            ma_up = up.ewm(com=periods - 1, adjust=True,
                           min_periods=periods).mean()
            ma_down = down.ewm(com=periods - 1, adjust=True,
                               min_periods=periods).mean()
        else:
            # Use simple moving average
            ma_up = up.rolling(window=periods, adjust=False).mean()
            ma_down = down.rolling(window=periods, adjust=False).mean()
        rsi = 100 - (100 / (1 + ma_up / ma_down))
        return rsi

    def macd(
            self,
            long_span: int = int(26),
            mid_span: int = int(12),
            short_span: int = int(9),
            adjust: bool = False,
    ) -> pd.DataFrame:
        """
        Get's MACD analysis and returns pandas DataFrame.
        :param long_span: int
        :param mid_span: int
        :param short_span: int
        :param adjust: bool
        :return: pd.DataFrame
        """
        df = self.__history
        macd = df['Close'].ewm(span=mid_span, adjust=adjust).mean() - \
            df['Close'].ewm(span=long_span, adjust=adjust).mean()
        signal = macd.ewm(span=short_span, adjust=adjust).mean()
        df_return = pd.DataFrame([])
        df_return['MACD'] = macd
        df_return['Signal'] = signal
        return df_return

    @property
    def history__(self) -> pd.DataFrame:
        """Get historical timeseries data."""
        return self.__history


class Trade:
    """Analyze expected return on trade(s)."""

    def __init__(self):
        """Instantiate the trade class."""

    def __call__(self, underlying: MonteCarlo, **kwargs):
        """Calling the class."""


@unique
class TransactionType(Enum):
    call = 'calls'
    put = 'puts'
    stock = 'stock'
    bond = 'bond'
    cash = 'cash'
