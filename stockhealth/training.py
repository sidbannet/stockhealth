#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth simulator
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

import numpy as np
from dataclasses import dataclass
from stockhealth.analyzer import TimeSeries as Stock
from scipy.stats import gaussian_kde, pearsonr
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
        self.__vol = df['Volatility'].mean() * np.sqrt(_NTD)
        self.__mew = data.mean() * _NTD
        self.__S = df['Close'][-1]
        self.heston_feature = None

    @dataclass
    class __StockFeature:
        kde1: gaussian_kde
        kde2: gaussian_kde
        reg1: object
        reg2: object
        std1: np.float
        std2: np.float
        correlation: np.float

    @property
    def history(self) -> dict:
        """Get historical trend data."""
        return {
            'roi': self.__mew,
            'volatility': self.__vol,
            'std': self.__std,
            'latest close': self.__S,
        }

    def extract_model_features(
            self,
    ) -> None:
        """
        Extract Heston model features using Approximate Bayesian Computing.
        """
        df = self.stock.history__
        number_of_days = self.number_of_days
        # Get the features of stochastic mean rate of return.
        y1 = - (
                (df['Risk free return']) * _NTD
        ).rolling(window=number_of_days).mean().diff(periods=-number_of_days)
        x1 = (
            (
                df['Risk free return'] - df['Risk free return'].mean()
            ) * _NTD
        ).rolling(window=number_of_days).mean()
        z1 = (df['Risk free return'] * _NTD).rolling(window=number_of_days).mean()
        kde1 = self.__extract_kde(x=x1, y=y1, n=number_of_days,)
        reg1 = self.__extract_regressor(x=x1, y=y1, n=number_of_days)
        std1 = np.nanstd(z1)
        # Get the features of stochastic volatility.
        y2 = (
            (df['Risk free return']) * _NTD
        ).shift(periods=-number_of_days).rolling(window=number_of_days).std() / (
            (df['Risk free return']) * _NTD
        ).rolling(window=number_of_days).std()
        x2 = (
            df['Risk free return'] * _NTD
        ).rolling(window=number_of_days).std() / (
            (df['Risk free return'] * _NTD).std()
        )
        z2 = (df['Risk free return'] * _NTD).rolling(window=number_of_days).std()
        kde2 = self.__extract_kde(
            x=vectorized_log(x2),
            y=vectorized_log(y2),
            n=number_of_days,
        )
        reg2 = self.__extract_regressor(
            x=vectorized_log(x2),
            y=vectorized_log(y2),
            n=number_of_days
        )
        std2 = np.nanstd(vectorized_log(z2))
        choice_array = np.logical_and(
            ~np.isnan(z1).values,
            ~np.isnan(z2).values
        )
        corr, _ = pearsonr(x=z1[choice_array], y=z2[choice_array])
        self.heston_feature = self.__StockFeature(
            kde1=kde1,
            kde2=kde2,
            reg1=reg1,
            reg2=reg2,
            std1=std1,
            std2=std2,
            correlation=corr,
        )
        self._trained = True

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
