#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth market data analyzer
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

from scipy.stats import norm
import numpy as np


# noinspection PyPep8Naming
class BlackScholes:
    """Black-Scholes model to determine fair European options price."""

    def __init__(
            self,
            S: np.float = np.nan,
            K: np.float = np.nan,
            T: np.float = np.nan,
            r: np.float = np.nan,
            q: np.float = np.float(0.0),
    ):
        """Instantiate the Black-Scholes model for options price.
        :type S: Spot price of the underlying asset
        :type K: Strike price
        :type T: Time to maturity (unit less fraction of one year)
        :type r: Risk free interest
        :type q: Annual dividend yeild
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.__d1 = lambda sigma: (
            (
                np.log(self.S / self.K) +
                (self.r - self.q + sigma ** 2 / 2.0) * self.T
            ) / (sigma * np.sqrt(self.T))
        )
        self.__d2 = lambda sigma: (
            (
                np.log(self.S / self.K) +
                (self.r - self.q + sigma ** 2 / 2.0) * self.T
            ) / (sigma * np.sqrt(self.T)) - (sigma * np.sqrt(self.T))
        )

    def _call_value(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """European Call option value given sigma."""
        S = self.S
        K = self.K
        T = self.T
        r = self.r
        return (
            S * norm.cdf(
                self.__d1(sigma=sigma)
            ) -
            K * norm.cdf(
                self.__d2(sigma=sigma)
            ) * np.exp(-r * T)
        )

    def _put_value(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """European Put option valuation given sigma."""
        S = self.S
        K = self.K
        T = self.T
        r = self.r
        return (
            np.exp(-r * T) * K * norm.cdf(
                - self.__d2(sigma=sigma)
            ) - S * norm.cdf(
                - self.__d1(sigma=sigma)
            )
        )
