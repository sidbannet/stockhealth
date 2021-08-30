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
