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
from datetime import date, datetime, timedelta
from stockhealth.model import European
from stockhealth.model \
    import StochasticVolatility as Model, Heston as HestonProcess
from stockhealth.training import Trends
from stockhealth.utilities.calendar import dt as date_difference
from stockhealth.utilities.graph import plot as probplt
from stockhealth.model import VolatilitySmile as Vsmile
from stockhealth.model import _NUMBER_OF_TRADING_DAYS_PER_YEAR as _NTD
from stockhealth.model import _NUMBER_OF_CALENDAR_DAYS_PER_YEAR as _NCD


# noinspection PyPep8Naming
class MonteCarlo:
    """
    Monte Carlo simulation of spot price given a stochastic model.

    Description
    -----------
    This class is used to simulate the spot price of a stock given a
    stochastic model. The simulation is done using the Euler-Maruyama
    scheme. The simulation is done in discrete time steps. The time
    steps are chosen to be the same as the number of trading days in a
    year. The simulation is done for a given number of days.

    Parameters
    ----------
    model : Model or HestonProcess
        The stochastic model to be used for the simulation.
    number_of_days : int
        The number of days for which the simulation is to be done.
    steps_in_days : int
        The number of steps in a day. The default is 1.
    stock_exchange_name : str
        The name of the stock exchange. The default is 'NYSE'.
    start_date : datetime
        The start date of the simulation. The default is today's date.

    Attributes
    ----------
    S : pd.DataFrame
        The simulated spot price.
    V : pd.DataFrame
        The simulated volatility.
    t_end : float
        The end time of the simulation.
    dt : float
        The time step of the simulation.

    Methods
    -------
    solve(day_trading=False)
        Solve the simulation.
    plot(plot_volatility=False)
        Plot the simulation.
    _stat(bins=1000)
        Get the CDF of the spot prices with time calculated.
    _name_dataframe_index(name='t')
        Name the dataframe index column.

    Examples
    --------
    >>> import numpy as np
    >>> from stockhealth.model import Heston
    >>> from stockhealth.simulation import MonteCarlo
    >>> mdl = Heston(
    ...     S0=100, r=0.05, q=0.01, kappa=1.5, theta=0.04, sigma=0.5,
    ...     rho=-0.75, v0=0.04, t=0, T=1,
    ... )
    >>> sim = MonteCarlo(
    ...     model=mdl, number_of_days=365, steps_in_days=1,
    ...     stock_exchange_name='NYSE', start_date=date(2021, 1, 1),
    ... )
    >>> sim.solve(day_trading=False)
    >>> sim.plot(plot_volatility=False)
    >>> sim._stat(bins=1000)
    """

    def __init__(
        self,
        model: Model or HestonProcess = None,
        number_of_days: int = np.nan,
        steps_in_days: int = int(1),
        stock_exchange_name: str = 'NYSE',
        start_date: datetime = datetime.today().date(),
    ):
        """Setup the simulation environment."""
        self._mdl = model
        self.t_end = float(number_of_days / _NTD)
        self.dt = float(steps_in_days / _NTD)
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

    @property
    def get_forecast__(self) -> pd.DataFrame:
        """Get the timeseries data of stock forecast."""
        assert self.__solved, "The simulation is not solved yet."
        return self.S


class MonteCarloWithTraining(MonteCarlo):
    """Sub-class of MonteCarlo which trains a model before simulations."""

    def __init__(
        self,
        trained_model: Trends = None,
        number_of_instances: int = int(10000),
        number_of_days: int = np.nan,
        steps_in_days: int = int(1),
        stock_exchange_name: str = 'NYSE',
        start_date: datetime = datetime.today().date(),
    ):
        """Instantiate the class."""
        stock_history = trained_model.stock.history__.loc[
            :np.datetime64(start_date)
        ]
        mew = trained_model.history['roi']
        price = stock_history['Close'].values[-1]
        sigma = trained_model.history['std']
        beta = float(0)
        kappa = float(0)
        epsilon = float(0)
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
        number_of_instances: int = int(10000),
        number_of_days: int = np.nan,
        steps_in_days: int = int(1),
        stock_exchange_name: str = 'NYSE',
        start_date: datetime = datetime.today().date(),
    ):
        """Instantiate the class."""
        historical_roi = trained_model.history['roi']
        historical_volatility = trained_model.history['perkinson volatility']
        stock_history = trained_model.stock.history__.loc[
            :np.datetime64(start_date)
        ]
        price = stock_history['Close'].values[-1]
        volatility = stock_history['Perkinson Volatility'].rolling(
            window=number_of_days,
        ).mean()[-1] * np.sqrt(_NTD)
        roi = stock_history['Risk free return'].rolling(
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

    Description
    -----------
    This class is used to predict the price of an option or a derivative
    future given an underlying stochastic simulation. The class is
    instantiated with an option and a simulation of the underlying
    asset. The class then solves for the price of the option or the
    derivative future.

    Parameters
    ----------
    option: European
        The option or derivative future to be solved for.
    simulation_of_underlying: MonteCarlo
        The simulation of the underlying asset.
    call_price: float
        The price of the call option. If not given, the class will
        solve for the call price.
    put_price: float
        The price of the put option. If not given, the class will
        solve for the put price.
    vsmile: Vsmile
        The volatility smile of the option. If not given, the class
        will solve for the volatility smile.

    Attributes
    ----------
    greeks: pd.DataFrame
        The greeks of the option.

    Methods
    -------
    solve()
        Solve for the option price or the volatility smile.
    plot()
        Plot the option price or the volatility smile.

    Examples
    --------
    >>> from stockhealth.simulation import (
    ...     MonteCarlo,
    ...     HestonProcess,
    ...     Derivative,
    ...     European,
    ...     Vsmile,
    ... )
    >>> from stockhealth.trends import Trends
    >>> from stockhealth.stock import Stock
    >>> from stockhealth.model import Model
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> # Create a stock object
    >>> stock = Stock(
    ...     ticker='AAPL',
    ...     start_date='2010-01-01',
    ...     end_date='2020-01-01',
    ...     stock_exchange_name='NASDAQ',
    ... )
    >>> # Create a model object
    >>> model = Model(
    ...     mew=0.05,
    ...     S0=stock.history__['Close'].values[-1],
    ...     sigma=0.2,
    ...     beta=0.5,
    ...     kappa=0.5,
    ...     epsilon=0.5,
    ...     number_of_instances=10000,
    ...     number_of_days=100,
    ...     steps_in_days=1,
    ...     stock_exchange_name='NASDAQ',
    ...     start_date='2020-01-01',
    ... )
    >>> # Create a trends object
    >>> trends = Trends(
    ...     stock=stock,
    ...     model=model,
    ...     number_of_days=100,
    ...     steps_in_days=1,
    ...     stock_exchange_name='NASDAQ',
    ...     start_date='2020-01-01',
    ... )
    >>> # Create a Monte Carlo object
    >>> mc = MonteCarlo(
    ...     model=model,
    ...     number_of_days=100,
    ...     steps_in_days=1,
    ...     stock_exchange_name='NASDAQ',
    ...     start_date='2020-01-01',
    ... )
    >>> # Create an option object
    >>> option = European(
    ...     strike=stock.history__['Close'].values[-1],
    ...     maturity=1,
    ...     option_type='call',
    ...     stock_exchange_name='NASDAQ',
    ...     start_date='2020-01-01',
    ... )
    >>> # Create a derivative object
    >>> derivative = Derivative(
    ...     option=option,
    ...     simulation_of_underlying=mc,
    ... )
    >>> # Solve for the option price
    >>> derivative.solve()
    >>> # Plot the option price
    >>> derivative.plot()
    >>> # Create a volatility smile object
    >>> vsmile = Vsmile(
    ...     option=option,
    ...     simulation_of_underlying=mc,
    ... )
    >>> # Solve for the volatility smile
    >>> vsmile.solve()
    >>> # Plot the volatility smile
    >>> vsmile.plot()
    """

    # noinspection PyPep8Naming,PyProtectedMember
    def __init__(
        self,
        option: European = None,
        simulation_of_underlying: MonteCarlo = None,
        call_price: float = np.nan,
        put_price: float = np.nan,
        vsmile: Vsmile = None,
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
        self.__initial_call_iv = option.sigma_call__(price=call_price)
        self.__initial_put_iv = option.sigma_put__(price=put_price)
        self.price = {
            'call': pd.DataFrame([]),
            'put': pd.DataFrame([]),
        }
        self.__call_price = np.vectorize(
            lambda S, K, T, r, q, sigma: option._call_price(
                S=S, K=K, T=T, r=r, q=q, sigma=sigma,
            )
        )
        self.__put_price = np.vectorize(
            lambda S, K, T, r, q, sigma: option._put_price(
                S=S, K=K, T=T, r=r, q=q, sigma=sigma,
            )
        )
        if vsmile is None:
            self.__vsmile = False
            self.__vsmile_slope = np.array([0.0, 0.0])
        else:
            self.__vsmile = True
            self.__vsmile_slope = vsmile.slopes
        self._options_forecast = pd.DataFrame([])
        self._solved = False

    @property
    def __iv_offset(
        self,
    ) -> np.array:
        """Get the volatility offset from volatility smile."""
        iv_offset = 0 * self.sim.V
        if not self.__vsmile:
            return iv_offset.values
        else:
            legs = [
                self.sim.S < self.option.K,
                self.sim.S >= self.option.K,
            ]
            for idx, leg in enumerate(legs):
                iv_offset[leg] =\
                    self.__vsmile_slope[idx] * (
                        self.sim.S[leg] - self.option.K
                    )
            return iv_offset.values

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
                now=now,
                reference=date.today() + timedelta(
                    int(self.option.T * _NCD)
                ),
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
        dividend_rate = np.full_like(interest_rate, fill_value=self.option.q)
        strike = np.full_like(interest_rate, fill_value=self.option.K)
        if self._solve_for_call:
            self.price['call'] = pd.DataFrame(
                data=self.__call_price(
                    S=self.sim.S.values,
                    K=strike,
                    sigma=(
                        self.sim.V.values * call_iv_multiplier
                        + self.__iv_offset
                    ),
                    T=time,
                    r=interest_rate,
                    q=dividend_rate,
                ),
                index=self.sim.S.index,
                columns=self.sim.S.columns,
            )
            call_price = self.price['call']
            call_price[call_price < 0.0] = 0.0
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
                    K=strike,
                    sigma=(
                        self.sim.V.values * put_iv_multiplier
                        + self.__iv_offset
                    ),
                    T=time,
                    r=interest_rate,
                    q=dividend_rate,
                ),
                index=self.sim.S.index,
                columns=self.sim.S.columns,
            )
            put_price = self.price['put']
            put_price[put_price < 0.0] = 0.0
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
        fig.suptitle(
            'Timeseries of future derivative price possibility statistics')
        return fig, axs

    @property
    def get_forecast__(
            self,
            number_of_shares_per_contract: int = int(100),
    ) -> pd.DataFrame:
        """Get timeseries forecast of the options value."""
        assert self._solved, "This simulation is not solved yet."
        return self._options_forecast * number_of_shares_per_contract
