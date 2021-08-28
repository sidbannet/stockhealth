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
from stockhealth.model import StochasticVolatility as Model


# noinspection PyPep8Naming
class MonteCarlo:
    """Monte Carlo simulation of spot price given a stochastic model."""

    def __init__(
            self,
            model: Model = None,
            time_end: np.float = np.nan,
            dt: np.float = np.float(1 / 365),
    ):
        """Setup the simulation environment."""
        self._mdl = model
        self.t_end = time_end
        self.dt = dt
        self.__solved = False
        self.S = pd.DataFrame(
            data=None,
            index=[],
            columns=list(range(model.S.size)),
        )
        self.V = pd.DataFrame(
            data=None,
            index=[],
            columns=list(range(model.volatility.size)),
        )
        self.__name_dataframe_index(name='t')
        # //todo: assert if the simulation is setup correctly

    def solve(self) -> None:
        """Solver for the simulation."""
        S = self.S.copy()
        V = self.V.copy()
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
        self.S = S.copy()
        self.V = V.copy()
        self.__name_dataframe_index(name='t')
        self._mdl.reset()
        self.__solved = True
        # //todo: reset the model after the solver

    def __name_dataframe_index(self, name: str = 't') -> None:
        """Name the dataframe index column."""
        self.S.index.name = self.V.index.name = name
