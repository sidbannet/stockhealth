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
from stockhealth.model import BlackScholes
from stockhealth.model import StochasticVolatility as Model, Heston as HestonProcess
from stockhealth.training import Trends
from stockhealth.utilities.calendar import dt as date_difference
from stockhealth.utilities.graph import plot as probplt
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

    def plot(self, plot_volatility: bool = False) -> tuple:
        """Plot timeseries statistics."""
        assert self.__solved, "This simulation is not solved yet."
        if not plot_volatility:
            fig, axs = probplt(self.S)
            axs.set_ylabel('Price')
        else:
            fig, axs = probplt(self.V)
            axs.set_ylabel('Volatility')
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
        mew = trained_model.history['roi']
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
        historical_roi = trained_model.history['roi']
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
        kappa_mew = trained_model.heston_feature.reg1.coef_[0] * \
            _NTD / trained_model.number_of_days
        kappa_y = trained_model.heston_feature.reg2.coef_[0] * \
            _NTD / trained_model.number_of_days
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


class Derivative:
    """
    Options or derivative future predictions given an underlying
    stochastic simulation is done.
    """

    # noinspection PyPep8Naming,PyProtectedMember
    def __init__(
            self,
            option: BlackScholes = None,
            simulation_of_underlying: MonteCarlo = None,
            call_price: np.float = np.nan,
            put_price: np.float = np.nan,
    ):
        """Instantiate the class."""
        self.option = option
        self.sim = simulation_of_underlying
        if call_price is np.nan:
            self._solve_for_call = False
        else:
            self._solve_for_call = True
        if put_price is np.nan:
            self._solve_for_put = False
        else:
            self._solve_for_put = True
        self.greeks = option.greeks(call_price=call_price, put_price=put_price)
        self.__underlying = {
            'S': pd.DataFrame([]),
            'V': pd.DataFrame([]),
        }
        self.__initial_call = call_price
        self.__initial_put = put_price
        self.__initial_call_iv = option._sigma_call(price=call_price)
        self.__initial_put_iv = option._sigma_put(price=put_price)
        self.price = {
            'call': pd.DataFrame([]),
            'put': pd.DataFrame([]),
        }
        self.__call_price = np.vectorize(
            lambda S, T, r, sigma: option._call_price(
                S=S, T=T, r=r, sigma=sigma,
            )
        )
        self.__put_price = np.vectorize(
            lambda S, T, r, sigma: option._put_price(
                S=S, T=T, r=r, sigma=sigma,
            )
        )
        self._options_forecast = pd.DataFrame([])
        self._solved = False

    def solve(self) -> None:
        """Solve for options future price forecast."""
        self.__underlying['S'] = pd.DataFrame(
            data=self.sim.S.values,
            index=self.sim.S.index,
            columns=self.sim.S.columns,
        )
        self.__underlying['V'] = pd.DataFrame(
            data=self.sim.V.values,
            index=self.sim.V.index,
            columns=self.sim.V.columns,
        )
        dt_from_reference = np.vectorize(
            lambda now: date_difference(
                now=now, reference=self.sim.S.index[-1],
            )
        )
        external_factors = pd.DataFrame(
            data=-dt_from_reference(self.sim.S.index),
            index=self.sim.S.index,
            columns=['time from expiry'],
        )
        external_factors['interest rate'] = self.option.r
        call_iv_multiplier = self.__initial_call_iv / self.sim.V.values[0][0]
        put_iv_multiplier = self.__initial_put_iv / self.sim.V.values[0][0]
        number_of_instances = self.sim.S.shape[1]
        time = np.transpose(
            np.tile(
                external_factors['time from expiry'].values,
                (number_of_instances, 1),
            )
        )
        interest_rate = np.transpose(
            np.tile(
                external_factors['interest rate'].values,
                (number_of_instances, 1),
            )
        )
        if self._solve_for_call:
            self.price['call'] = pd.DataFrame(
                data=self.__call_price(
                    S=self.sim.S.values,
                    sigma=self.sim.V.values * call_iv_multiplier,
                    T=time,
                    r=interest_rate,
                ),
                index=self.sim.S.index,
                columns=self.sim.S.columns,
            )
        else:
            self.price['call'] = pd.DataFrame(
                data=np.full_like(self.sim.S.values, fill_value=0.0),
                index=self.sim.S.index,
                columns=self.sim.S.columns,
            )
        if self._solve_for_put:
            self.price['put'] = pd.DataFrame(
                data=self.__put_price(
                    S=self.sim.S.values,
                    sigma=self.sim.V.values * put_iv_multiplier,
                    T=time,
                    r=interest_rate,
                ),
                index=self.sim.S.index,
                columns=self.sim.S.columns,
            )
        else:
            self.price['put'] = pd.DataFrame(
                data=np.full_like(self.sim.S.values, fill_value=0.0),
                index=self.sim.S.index,
                columns=self.sim.S.columns,
            )
        self._options_forecast = self.price['call'] + self.price['put']
        self._solved = True

    def plot(self) -> tuple:
        """Timeseries plot with uncertainty bands."""
        assert self._solved, "The Derivative futures is not simulated yet."
        fig = plt.figure('Timeseries of options price statistics')
        axs = fig.subplots(nrows=2, ncols=1, sharex=True)
        fig, axs[0] = probplt(self.price['call'], fig=fig, axs=axs[0])
        fig, axs[1] = probplt(self.price['put'], fig=fig, axs=axs[1])
        axs[0].set_title('Call')
        axs[1].set_title('Put')
        _ = [ax.set_ylabel('Price') for ax in axs.flat]
        fig.suptitle('Timeseries of future derivative price possibility statistics')
        return fig, axs
