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
        self.solution = pd.DataFrame(data=None)
        # //todo: assert if the simulation is setup correctly

    def solve(self) -> None:
        """Solver for the simulation."""
        t = 0
        time = [t, ]
        S = [self._mdl.S, ]
        while t <= self.t_end:
            t += self.dt
            self._mdl.update(dt=self.dt)
            time.append(t)
            S.append(self._mdl.S)
        self.solution = pd.DataFrame(
            {
                't': time,
                'S': S,
            }
        ).set_index('t')
        self.__solved = True
        # //todo: reset the model after the solver
