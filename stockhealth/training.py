#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth simulator
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

import numpy as np
from stockhealth.analyzer import TimeSeries as Stock
from stockhealth.model import _NUMBER_OF_TRADING_DAYS_PER_YEAR as _NTD


class Trends:
    """Get historical trends given a stock."""

    def __init__(
            self,
            stock: Stock = None,
            number_of_days: np.int = np.nan,
    ):
        """Instantiate the class."""
        self.stock = stock
        self._trained = False
        df = stock.history__
        if number_of_days is not np.nan and number_of_days >= _NTD:
            data = df['Risk free return'][-number_of_days:]
        else:
            data = df['Risk free return']
        self.__std = data.std() * np.sqrt(_NTD)
        self.__mew = data.mean() * _NTD
        self.__S = df['Close'][-1]

    @property
    def history(self) -> dict:
        """Get historical trend data."""
        return {
            'mew': self.__mew,
            'std': self.__std,
            'latest close': self.__S,
        }
