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
from scipy.stats import gaussian_kde
from sklearn.linear_model import HuberRegressor as Regressor
from stockhealth.model import _NUMBER_OF_TRADING_DAYS_PER_YEAR as _NTD

vectorized_log = np.vectorize(np.log)


class Trends:
    """Get historical trends given a stock."""

    def __init__(
            self,
            stock: Stock = None,
            number_of_days: np.int = int(1),
    ):
        """Instantiate the class."""
        self.stock = stock
        self._trained = False
        df = stock.history__
        if number_of_days is not np.nan and number_of_days >= _NTD:
            data = df['Risk free return'][-number_of_days:]
            self.number_of_days = number_of_days
        else:
            data = df['Risk free return']
            self.number_of_days = int(30)
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

    def extract_model_features(
            self,
    ) -> tuple:
        """
        Extract Heston model features using Approximate Bayesian Computing.
        """
        df = self.stock.history__
        number_of_days = self.number_of_days
        # Get the features of stochastic mean rate of return.
        y = - (
                (df['Risk free return']) * _NTD * 100
        ).rolling(window=number_of_days).mean().diff(periods=-number_of_days)
        x = (
            (
                df['Risk free return'] - df['Risk free return'].mean()
            ) * _NTD * 100
        ).rolling(window=number_of_days).mean()
        z = (df['Risk free return'] * _NTD * 100).rolling(window=number_of_days).mean()
        kde1 = self.__extract_kde(x=x, y=y, n=number_of_days,)
        reg1 = self.__extract_regressor(x=x, y=y, n=number_of_days)
        std1 = np.nanstd(z)
        # Get the features of stochastic volatility.
        y = (
            (df['Risk free return']) * _NTD * 100
        ).shift(periods=number_of_days).rolling(window=number_of_days).std() / (
            (df['Risk free return']) * _NTD * 100
        ).rolling(window=number_of_days).std()
        x = (
            df['Risk free return'] * _NTD * 100
        ).rolling(window=number_of_days).std() / (
            df['Risk free return'].std() * _NTD * 100
        )
        z = (df['Risk free return'] * _NTD * 100).rolling(window=number_of_days).std()
        kde2 = self.__extract_kde(
            x=vectorized_log(x),
            y=vectorized_log(y),
            n=2*number_of_days,
        )
        reg2 = self.__extract_regressor(
            x=vectorized_log(x),
            y=vectorized_log(y),
            n=2*number_of_days
        )
        std2 = np.nanstd(vectorized_log(z))
        return kde1, kde2, reg1, reg2, std1, std2

    @staticmethod
    def __extract_kde(
        x: np.array,
        y: np.array,
        n: int,
    ) -> gaussian_kde:
        """Extract 2D kernel density function."""
        values = np.vstack([x[n - 1: -n], y[n - 1: -n]])
        return gaussian_kde(values)

    @staticmethod
    def __extract_regressor(
        x: np.array,
        y: np.array,
        n: int,
        intercept: bool = False,
    ) -> object:
        """Extract Huber regressor from x, y values."""
        return Regressor(
            fit_intercept=intercept,
        ).fit(
            X=np.array(x[n - 1: -n]).reshape(-1, 1),
            y=np.array(y[n - 1: -n])
        )
