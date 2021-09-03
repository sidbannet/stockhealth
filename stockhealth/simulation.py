#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth simulator
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pandas_market_calendars import get_calendar as market_calendar
from datetime import datetime, timedelta
from stockhealth.model import StochasticVolatility as Model


# noinspection PyPep8Naming
class MonteCarlo:
    """Monte Carlo simulation of spot price given a stochastic model."""

    def __init__(
            self,
            model: Model = None,
            number_of_days: np.int = np.nan,
            steps_in_days: np.int = np.int(1),
            stock_exchange_name: str = 'NYSE',
            start_date: datetime = datetime.today().date(),
    ):
        """Setup the simulation environment."""
        self._mdl = model
        self.t_end = np.float(number_of_days / 365)
        self.dt = np.float(steps_in_days / 365)
        self.__steps_in_days = steps_in_days
        self.__solved = False
        self.__start_date = start_date
        self.__number_of_days = number_of_days
        self.__exchange_calendar = market_calendar(stock_exchange_name)
        self.S = pd.DataFrame(
            data=self._mdl.S, columns=[self._mdl.t],
        ).T
        self.V = pd.DataFrame(
            data=self._mdl.volatility, columns=[self._mdl.t],
        ).T
        self.__name_dataframe_index(name='t')
        # //todo: assert if the simulation is setup correctly

    def solve(self, day_trading: bool = False) -> None:
        """Solver for the simulation."""
        assert not self.__solved, "This simulation is already done!!!"
        S, V = self.S.copy(), self.V.copy()
        while self._mdl.t <= self.t_end:
            self._mdl.update(dt=self.dt)
            S = S.append(
                pd.DataFrame(
                    data=self._mdl.S, columns=[self._mdl.t]
                ).T,
            )
            V = V.append(
                pd.DataFrame(
                    data=self._mdl.volatility, columns=[self._mdl.t]
                ).T,
            )
        self.S, self.V = S.copy(), V.copy()
        if not day_trading:
            scheduled_days = self.__exchange_calendar.schedule(
                start_date=self.__start_date,
                end_date=self.__start_date + timedelta(
                    days=2 * len(self.S) + 50),
            )
            dates = scheduled_days.index[:self.S.__len__()].date
            self.S.index = self.V.index = dates
            self.__name_dataframe_index(name='Date')
        else:
            self.__name_dataframe_index(name='t')
        self._mdl.reset()
        self.__solved = True

    def __name_dataframe_index(self, name: str = 't') -> None:
        """Name the dataframe index column."""
        self.S.index.name = self.V.index.name = name

    def plot(self) -> tuple:
        """Plot timeseries statistics."""
        assert self.__solved, "This simulation is not solved yet."
        mean, std = self.S.mean(axis='columns'), self.S.std(axis='columns')
        df_stat = pd.DataFrame(
            {
                '-3 sigma': mean - 3 * std,
                '-2 sigma': mean - 2 * std,
                '-1 sigma': mean - 1 * std,
                '+1 sigma': mean + 1 * std,
                '+2 sigma': mean + 2 * std,
                '+3 sigma': mean + 3 * std,
            }
        )
        fig = plt.figure('Timeseries of statistics')
        axs = fig.subplots(nrows=1, ncols=1)
        axs.fill_between(
            x=df_stat.index, y1=df_stat['-3 sigma'], y2=df_stat['+3 sigma'],
            where=df_stat['+3 sigma'] > df_stat['-3 sigma'],
            facecolor='green', alpha=0.2, interpolate=True,
        )
        axs.fill_between(
            x=df_stat.index.to_list(),
            y1=df_stat['-2 sigma'],
            y2=df_stat['+2 sigma'],
            where=df_stat['+2 sigma'] > df_stat['-2 sigma'],
            facecolor='green', alpha=0.4, interpolate=True,
        )
        axs.fill_between(
            x=df_stat.index.to_list(),
            y1=df_stat['-1 sigma'],
            y2=df_stat['+1 sigma'],
            where=df_stat['+1 sigma'] > df_stat['-1 sigma'],
            facecolor='green', alpha=0.6, interpolate=True,
        )
        axs.grid(True)
        axs.legend(['3 sigma', '2 sigma', '1 sigma'])
        axs.set_title('Sigma spreads')
        axs.set_ylabel('Price')
        axs.set_xlabel('Time')
        fig.suptitle('Timeseries of future spot price possibility statistics')
        fig.autofmt_xdate(rotation=45)
        return fig, axs

    def _stat(self, bins: int = int(1000)) -> pd.DataFrame:
        """Get PDF and CDF of the spo prices with time."""
        assert self.__solved, "This simulation is not solved yet."
        x = np.linspace(self.S.min().min(), self.S.max().max(), bins)
        cdf = pd.DataFrame(
            {
                'S': x,
            }
        )
        for k, v in self.S.iterrows():
            if v.values.std() != 0:
                kde = gaussian_kde(v.values)
                kde_cdf = kde.evaluate(x).cumsum()
                kde_cdf /= kde_cdf.max()
                cdf[k] = kde_cdf
        return cdf.set_index('S')
