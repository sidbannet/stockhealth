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
from sklearn.linear_model import LinearRegression as regression
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
            (df['Risk free return'] - df['Risk free return'].mean()) * _NTD * 100
        ).rolling(window=number_of_days).mean()
        kde1 = self.__extract_kde(x=x, y=y, n=number_of_days,)
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
        kde2 = self.__extract_kde(
            x=vectorized_log(x),
            y=vectorized_log(y),
            n=2*number_of_days,
        )
        return kde1, kde2

    def extract_coeffs(
        self,
    ) -> tuple:
        """
        Extract Heston model coefficients using regression fits.
        """
        df = self.stock.history__
        number_of_days = self.number_of_days
        kde_mean, kde_vol = self.extract_model_features()
        num_resolution = 1000
        # Get coefficients for stochastic mean return 
        y = - (
            (df['Risk free return']) * _NTD * 100
        ).rolling(window=number_of_days).mean().diff(periods=-number_of_days)
        x = (
            (df['Risk free return'] - df['Risk free return'].mean()) * _NTD * 100
        ).rolling(window=number_of_days).mean()
        yy = np.linspace(start=y.min(), stop=y.max(), num=num_resolution) 
        x_ = np.linspace(start=x.min(), stop=x.max(), num=num_resolution)
        get_mean_mean = lambda x: self.__extract_norm_mean_from_kde(
            yy=yy, x=x, kde=kde_mean,
        )
        get_std_mean = lambda x: self.__extract_norm_sigma_from_kde(
            yy=yy, x=x, kde=kde_mean,
        )
        mean = np.vectorize(get_mean_mean)(x_)
        std = np.vectorize(get_std_mean)(x_)
        reg_mean = regression(fit_intercept=False).fit(X=x_, y=mean)
        # Get coefficients for stochastic volatility
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
        yy = np.linspace(start=y.min(), stop=y.max(), num=num_resolution)
        x_ = np.linspace(start=x.min(), stop=x.max(), num=num_resolution)
        get_mean_std = lambda x: self.__extract_norm_mean_from_kde(
            yy=yy, x=x, kde=kde_vol,
        )
        get_std_std = lambda x: self.__extract_norm_sigma_from_kde(
            yy=yy, x=x, kde=kde_vol,
        )
        mean = np.vectorize(get_mean_std)(x_)
        std = np.vectorize(get_std_std)(x_)
        reg_std = regression(fit_intercept=False).fit(X=x_, y=mean)
        return reg_mean, reg_std

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
    def __extract_norm_mean_from_kde(
        yy: np.array,
        x: np.float,
        kde: gaussian_kde,
    ) -> np.float:
        """Extract the mean from kde."""
        positions = np.vstack([np.full_like(yy, x), yy])
        cdf = kde.evaluate(positions).T.cumsum()
        cdf /= cdf.max()
        return yy[np.argwhere(cdf >= 0.5)[0][0]]

    @staticmethod
    def __extract_norm_sigma_from_kde(
        yy: np.array,
        x: np.float,
        kde: gaussian_kde,
    ) -> np.float:
        """Extract the std div from kde."""
        positions = np.vstack([np.full_like(yy, x), yy])
        cdf = kde.evaluate(positions).T.cumsum()
        cdf /= cdf.max()
        return yy[
            np.argwhere(cdf >= 0.841)[0][0]
        ] - yy[
            np.argwhere(cdf <= 0.159)[-1][0]
        ]
