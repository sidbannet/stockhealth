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
from stockhealth.model import StochasticVolatility as Model, Heston as HestonProcess
from stockhealth.training import Trends
from stockhealth.model import _NUMBER_OF_TRADING_DAYS_PER_YEAR as _NTD


# noinspection PyPep8Naming
class MonteCarlo:
    """Monte Carlo simulation of spot price given a stochastic model."""

    def __init__(
            self,
            model: Model or HestonProcess = None,
            number_of_days: np.int = np.nan,
            steps_in_days: np.int = np.int(1),
            stock_exchange_name: str = 'NYSE',
            start_date: datetime = datetime.today().date(),
    ):
        """Setup the simulation environment."""
        self._mdl = model
        self.t_end = np.float(number_of_days / _NTD)
        self.dt = np.float(steps_in_days / _NTD)
        self.__steps_in_days = steps_in_days
        self.__solved = False
        self.__start_date = start_date
        self.__number_of_days = number_of_days
        self.__exchange_calendar = market_calendar(stock_exchange_name)
        self._cdf = {}
        self.__cdf_calculated = False
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
        v = self.S.values.copy()
        v.sort(axis=1)
        df_prob_ = pd.DataFrame(data=v, index=self.S.index).T
        df_prob_['p'] = df_prob_.index / df_prob_.index.max()
        df_prob_ = df_prob_.set_index('p')
        df_prob__ = pd.DataFrame(
            columns=df_prob_.columns,
            index=[0.00135, 0.02275, 0.15865, 0.5, 0.84135, 0.97725, 0.99865],
        )
        df_prob = pd.concat(
            [df_prob_, df_prob__],
        ).sort_index().interpolate(axis=0).T
        fig = plt.figure('Timeseries of statistics')
        axs = fig.subplots(nrows=2, ncols=1, sharex=True,)
        axs[0].fill_between(
            x=df_stat.index, y1=df_stat['-3 sigma'], y2=df_stat['+3 sigma'],
            where=df_stat['+3 sigma'] > df_stat['-3 sigma'],
            facecolor='green', alpha=0.2, interpolate=True,
        )
        axs[0].fill_between(
            x=df_stat.index.to_list(),
            y1=df_stat['-2 sigma'],
            y2=df_stat['+2 sigma'],
            where=df_stat['+2 sigma'] > df_stat['-2 sigma'],
            facecolor='green', alpha=0.4, interpolate=True,
        )
        axs[0].fill_between(
            x=df_stat.index.to_list(),
            y1=df_stat['-1 sigma'],
            y2=df_stat['+1 sigma'],
            where=df_stat['+1 sigma'] > df_stat['-1 sigma'],
            facecolor='green', alpha=0.6, interpolate=True,
        )
        mean.plot(ax=axs[0], label='mean', style='--', color='k',)
        axs[1].fill_between(
            x=df_prob.index, y1=df_prob[0.00135], y2=df_prob[0.99865],
            where=df_prob[0.99865] > df_prob[0.00135],
            facecolor='blue', alpha=0.2, interpolate=True,
        )
        axs[1].fill_between(
            x=df_prob.index, y1=df_prob[0.02275], y2=df_prob[0.97725],
            where=df_prob[0.97725] > df_prob[0.02275],
            facecolor='blue', alpha=0.4, interpolate=True,
        )
        axs[1].fill_between(
            x=df_prob.index, y1=df_prob[0.15865], y2=df_prob[0.84135],
            where=df_prob[0.84135] > df_prob[0.15865],
            facecolor='blue', alpha=0.6, interpolate=True,
        )
        df_prob[0.5].plot(ax=axs[1], label='median', style='-.', color='k',)
        _ = [ax.grid(True) for ax in axs.flat]
        axs[0].legend(['mean', '3 sigma', '2 sigma', '1 sigma'])
        axs[1].legend(['median', '99.74 %', '95.45 %', '68.27 %'])
        axs[0].set_title('Sigma spreads')
        axs[1].set_title('Confidence Interval')
        _ = [ax.set_ylabel('Price') for ax in axs.flat]
        axs[-1].set_xlabel('Time')
        fig.suptitle('Timeseries of future spot price possibility statistics')
        fig.autofmt_xdate(rotation=45)
        return fig, axs

    def _stat(self, bins: int = int(1000)) -> None:
        """Get CDF of the spot prices with time calculated."""
        assert self.__solved, "This simulation is not solved yet."
        if self.__cdf_calculated:
            return
        x = np.linspace(self.S.min().min(), self.S.max().max(), bins)
        cdf = pd.DataFrame(
            {
                'S': x,
            }
        )
        for k, v in self.S.iterrows():
            if v.values.std() > 1e-12:
                kde = gaussian_kde(v.values)
                kde_cdf = kde.evaluate(x).cumsum()
                kde_cdf /= kde_cdf.max()
                cdf[k] = kde_cdf
        self._cdf = {
            'S': cdf.set_index('S'),
        }
        self.__cdf_calculated = True


class MonteCarloWithTraining(MonteCarlo):
    """Sub-class of MonteCarlo which trains a model before simulations."""

    def __init__(
            self,
            trained_model: Trends = None,
            number_of_instances: np.int = np.int(10000),
            number_of_days: np.int = np.nan,
            steps_in_days: np.int = np.int(1),
            stock_exchange_name: str = 'NYSE',
            start_date: datetime = datetime.today().date(),
    ):
        """Instantiate the class."""
        mew = trained_model.history['mew']
        price = trained_model.history['latest close']
        sigma = trained_model.history['std']
        beta = np.float(0)
        kappa = np.float(0)
        epsilon = np.float(0)
        super().__init__(
            model=Model(
                mew=mew,
                S0=price,
                sigma=sigma,
                beta=beta,
                kappa=kappa,
                epsilon=epsilon,
                number_of_instances=number_of_instances,
            ),
            number_of_days=number_of_days,
            steps_in_days=steps_in_days,
            stock_exchange_name=stock_exchange_name,
            start_date=start_date,
        )


class MonteCarlosWithHeston(MonteCarlo):
    """Sub-class of MonteCarlo with trained Heston process."""

    def __init__(
            self,
            trained_model: Trends,
            number_of_instances: np.int = np.int(10000),
            number_of_days: np.int = np.nan,
            steps_in_days: np.int = np.int(1),
            stock_exchange_name: str = 'NYSE',
            start_date: datetime = datetime.today().date(),
    ):
        """Instantiate the class."""
        historical_roi = trained_model.history['mew']
        historical_volatility = trained_model.history['std']
        price = trained_model.history['latest close']
        volatility = trained_model.stock.history__['Volatility'].rolling(
            window=number_of_days,
        ).mean()[-1] * np.sqrt(_NTD)
        roi = trained_model.stock.history__['Risk free return'].rolling(
            window=number_of_days,
        ).mean()[-1] * _NTD
        # noinspection PyProtectedMember
        if not trained_model._trained:
            trained_model.extract_model_features()
        rho = trained_model.heston_feature.correlation
        kappa_mew = trained_model.heston_feature.reg1.coef_[0]
        kappa_y = trained_model.heston_feature.reg2.coef_[0]
        sigma_mew = trained_model.heston_feature.std1
        sigma_y = trained_model.heston_feature.std2
        super().__init__(
            model=HestonProcess(
                S0=price,
                mew0=roi,
                V0=volatility,
                historical_roi=historical_roi,
                historical_volatility=historical_volatility,
                mean_reversion_roi=kappa_mew,
                mean_reversion_log_volatility=kappa_y,
                sigma_mew=sigma_mew,
                sigma_y=sigma_y,
                correlation=rho,
                number_of_instances=number_of_instances,
            ),
            number_of_days=number_of_days,
            steps_in_days=steps_in_days,
            stock_exchange_name=stock_exchange_name,
            start_date=start_date,
        )
