#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth market data analyzer
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import pytz
from enum import Enum, unique
from collections import namedtuple
import yfinance as yf
from stockhealth.model import BlackScholes as Bs
from stockhealth.model import _NUMBER_OF_CALENDAR_DAYS_PER_YEAR as _NCD

Transaction = namedtuple('forecast', ['simulation', 'amount'])


@unique
class TransactionType(Enum):
    call = 'calls'
    put = 'puts'
    stock = 'stock'
    bond = 'bond'
    cash = 'cash'


@unique
class OptionsGreekType(Enum):
    delta = 'delta'
    gamma = 'gamma'
    vega = 'vega'
    theta = 'theta'
    rho = 'rho'
    intrinsic = 'intrinsic'
    extrinsic = 'extrinsic'


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
        df['Real Worth'].values[0] = df['Open'].values[0]
        df__ = pd.DataFrame([])
        df__['High'] = df['High']
        df__['Low'] = df['Low']
        df__['last price'] = df['Close'].shift(periods=1)
        df__['last price'].values[0] = df['Open'].values[0]
        df__['Real High'] = df__[['last price', 'High']].max(axis=1)
        df__['Real Low'] = df__[['last price', 'Low']].min(axis=1)
        df['Risk free return'] = df['Real Worth'].diff(periods=1) / \
            df['Real Worth'].shift(periods=1)
        df['Risk free return'].values[0] = float(0.0)
        df['Volatility'] = (
            df__['Real High'] - df__['Real Low']
        ) / df__['last price']

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

    @property
    def current__(self) -> float:
        """Get the latest stock price."""
        return self.__ticker.history(
            period='1d', interval='1m',
        )['Close'].values[-1]

    @property
    def dividend_yield(self) -> np.float:
        """Get dividend yield."""
        date_a_year_ago = pytz.utc.localize(
            datetime.combine(
                date.today() - timedelta(_NCD),
                datetime.min.time()
            )
        )
        return self.__ticker.dividends.loc[
            date_a_year_ago:
        ].sum() / self.history__['Close'].iloc[-1]

    # noinspection PyUnresolvedReferences,PyPep8Naming
    @staticmethod
    def __get_greeks(
        type_of_transaction: TransactionType,
        times: np.array,
        prices: np.array,
        strikes: np.array,
        sigmas: np.array,
        S: np.float,
        r: np.float,
        q: np.float = 0.0,
    ) -> dict:
        """Get greeks from the options chain."""
        mdl = Bs()
        greeks = {}
        rs = np.full_like(prices, fill_value=r)
        qs = np.full_like(prices, fill_value=q)
        Ss = np.full_like(prices, fill_value=S)

        # noinspection PyPep8Naming
        def get_greeks__(
                fns_delta,
                fns_gamma,
                fns_theta,
                fns_rho,
                fns_vega,
                fns_sigma,
                fns_fair_market_price,
                Ss__: np.array,
                Ks__: np.array,
                Ts__: np.array,
                rs__: np.array,
                qs__: np.array,
                sigmas__: np.array,
                prices__: np.array,
        ) -> dict:
            """Get the Greeks given the functions."""
            deltas_ = np.vectorize(
                lambda Ss_, Ks_, Ts_, rs_, qs_, sigmas_, : fns_delta(
                    S=Ss_, K=Ks_, T=Ts_, r=rs_, q=qs_, sigma=sigmas_,
                )
            )(
                Ss_=Ss__,
                Ks_=Ks__,
                Ts_=Ts__,
                rs_=rs__,
                qs_=qs__,
                sigmas_=sigmas__,
            )
            gammas_ = np.vectorize(
                lambda Ss_, Ks_, Ts_, rs_, qs_, sigmas_, : fns_gamma(
                    S=Ss_, K=Ks_, T=Ts_, r=rs_, q=qs_, sigma=sigmas_,
                )
            )(
                Ss_=Ss__,
                Ks_=Ks__,
                Ts_=Ts__,
                rs_=rs__,
                qs_=qs__,
                sigmas_=sigmas__,
            )
            thetas_ = np.vectorize(
                lambda Ss_, Ks_, Ts_, rs_, qs_, sigmas_, : fns_theta(
                    S=Ss_, K=Ks_, T=Ts_, r=rs_, q=qs_, sigma=sigmas_,
                )
            )(
                Ss_=Ss__,
                Ks_=Ks__,
                Ts_=Ts__,
                rs_=rs__,
                qs_=qs__,
                sigmas_=sigmas__,
            )
            rhos_ = np.vectorize(
                lambda Ss_, Ks_, Ts_, rs_, qs_, sigmas_, : fns_rho(
                    S=Ss_, K=Ks_, T=Ts_, r=rs_, q=qs_, sigma=sigmas_,
                )
            )(
                Ss_=Ss__,
                Ks_=Ks__,
                Ts_=Ts__,
                rs_=rs__,
                qs_=qs__,
                sigmas_=sigmas__,
            )
            vegas_ = np.vectorize(
                lambda Ss_, Ks_, Ts_, rs_, qs_, sigmas_, : fns_vega(
                    S=Ss_, K=Ks_, T=Ts_, r=rs_, q=qs_, sigma=sigmas_,
                )
            )(
                Ss_=Ss__,
                Ks_=Ks__,
                Ts_=Ts__,
                rs_=rs__,
                qs_=qs__,
                sigmas_=sigmas__,
            )
            volatility_ = np.vectorize(
                lambda Ss_, Ks_, Ts_, rs_, qs_, prices_, : fns_sigma(
                    S=Ss_, K=Ks_, T=Ts_, r=rs_, q=qs_, price=prices_,
                )
            )(
                Ss_=Ss__,
                Ks_=Ks__,
                Ts_=Ts__,
                rs_=rs__,
                qs_=qs__,
                prices_=prices__,
            )
            fair_market_prices_ = np.vectorize(
                lambda Ss_, Ks_, Ts_,
                rs_, qs_, sigmas_, : fns_fair_market_price(
                    S=Ss_, K=Ks_, T=Ts_, r=rs_, q=qs_, sigma=sigmas_,
                )
            )(
                Ss_=Ss__,
                Ks_=Ks__,
                Ts_=Ts__,
                rs_=rs__,
                qs_=qs__,
                sigmas_=sigmas__,
            )
            return {
                'delta': deltas_,
                'gamma': gammas_,
                'theta': thetas_,
                'rho': rhos_,
                'vega': vegas_,
                'sigma': volatility_,
                'theo price': fair_market_prices_,
            }

        if type_of_transaction.value == 'calls':
            greeks = get_greeks__(
                fns_delta=mdl._call_delta,
                fns_gamma=mdl._gamma,
                fns_theta=mdl._call_theta,
                fns_vega=mdl._vega,
                fns_rho=mdl._call_rho,
                fns_sigma=mdl._sigma_call,
                fns_fair_market_price=mdl._call_price,
                Ss__=Ss, Ks__=strikes, Ts__=times,
                rs__=rs, qs__=qs, sigmas__=sigmas,
                prices__=prices,
            )
        elif type_of_transaction.value == 'puts':
            greeks = get_greeks__(
                fns_delta=mdl._call_delta,
                fns_gamma=mdl._gamma,
                fns_theta=mdl._call_theta,
                fns_vega=mdl._vega,
                fns_rho=mdl._call_rho,
                fns_sigma=mdl._sigma_call,
                fns_fair_market_price=mdl._put_price,
                Ss__=Ss, Ks__=strikes, Ts__=times,
                rs__=rs, qs__=qs, sigmas__=sigmas,
                prices__=prices,
            )
        return greeks

    def options_chain__(
            self,
            expiry_date: date,
            type_of_transaction: TransactionType,
            interest_rate: np.float = 0.01,
            greek_on: bool = False,
    ) -> pd.DataFrame:
        """Get options chain properties."""
        chain = self.__ticker.option_chain(
            date=expiry_date.strftime('%Y-%m-%d')
        ).__getattribute__(type_of_transaction.value)
        number_of_seconds_in_calendar = _NCD * 60 * 60 * 24
        dt = (
            pd.to_datetime(
                expiry_date.strftime('%Y-%m-%d') + 'T23:59:59.00'
            ).tz_localize(tz=None).to_datetime64() -
            pd.to_datetime(chain['lastTradeDate'].values)
        ).astype('timedelta64[s]').astype('float') / np.timedelta64(
            number_of_seconds_in_calendar.__int__(), 's'
        ).astype('float')
        chain['time to expiry in calendar year'] = dt.values
        if greek_on:
            greeks = self.__get_greeks(
                times=dt.values,
                prices=chain['lastPrice'].values,
                strikes=chain['strike'].values,
                sigmas=chain['impliedVolatility'].values,
                S=self.__ticker.history(
                    period='1d', interval='1m'
                )['Close'].values[-1],
                r=interest_rate,
                q=self.dividend_yield,
                type_of_transaction=type_of_transaction,
            )
            return chain.join(pd.DataFrame(greeks)).set_index('strike')
        else:
            return chain.set_index('strike')


class Trade:
    """Analyze expected return on trade(s)."""

    def __init__(
            self,
            base_transaction: Transaction,
            *args: Transaction,
    ):
        """Instantiate the trade class."""
        price = base_transaction.simulation.get_forecast__ \
            * base_transaction.amount
        for arg in args:
            price += arg.simulation.get_forecast__ * arg.amount
        self.price = price
