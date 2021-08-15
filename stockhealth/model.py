#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth market data analyzer
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np


# noinspection PyPep8Naming
class BlackScholes:
    """Black-Scholes model to determine fair European options price."""

    def __init__(
            self,
            S: float = np.nan,
            K: float = np.nan,
            T: float = np.nan,
            r: float = np.nan,
            sigma: float = np.nan,
            q: float = float(0.0),
    ):
        """Instantiate the Black-Scholes model for options price.
        :type S: Spot price of the underlying asset
        :type K: Strike price
        :type T: Time to maturity (unit less fraction of one year)
        :type r: Risk free interest
        :type sigma: Volaility of the underlying asset
        :type q: Annual dividend yeild
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        d1 = (
            log(S / K) + (r + sigma ** 2 / 2.0) * T / (sigma * sqrt(T))
        )
        d2 = d1 - sigma * sqrt(T)
        self.__d1 = d1
        self.__d2 = d2
        self.__call = exp(-r * T) * (
            exp(r * T) * S * norm.cdf(d1) - K * norm.cdf(d2)
        )
        self.__put = exp(-r * T) * (
            K * norm.cdf(-d2) - exp(r * T) * S * norm.cdf(-d1)
        )
        call_delta = norm.cdf(d1) * exp(-q * T)
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T)) * exp(-q * T)
        vega = 0.01 * (S * norm.pdf(d1) * sqrt(T)) * exp(-q * T)
        call_theta = 0.01 * (
                - exp(-q * T) * (S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) -
                r * K * exp(-r * T) * norm.cdf(d2) +
                q * S * exp(-q * T) * norm.cdf(d1)
        )
        call_rho = 0.01 * (K * T * exp(-r * T) * norm.cdf(d2))
        put_delta = - norm.cdf(-d1) * exp(-q * T)
        put_theta = 0.01 * (
            - exp(-q * T) * (S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) +
            r * K * exp(- r * T) * norm.cdf(-d2) -
            q * S * exp(-q * T) * norm.cdf(-d1)
        )
        put_rho = 0.01 * (-K * T * exp(-r * T) * norm.cdf(-d2))
        self.__greeks = {
            'call': {
                'delta': call_delta,
                'gamma': gamma,
                'theta': call_theta,
                'vega': vega,
                'rho': call_rho,
            },
            'put': {
                'delta': put_delta,
                'gamma': gamma,
                'theta': put_theta,
                'vega': vega,
                'rho': put_rho,
            },
        }

    @property
    def greeks_(self) -> dict:
        return self.__greeks

    @property
    def value_(self) -> dict:
        return {
            'call': self.__call,
            'put': self.__put,
        }

    def call_implied_volatility__(
            self,
            price: np.float = np.nan,
    ) -> np.float:
        """Expected future volatility of underlying within time to maturity."""
        sigma = 0.001
        while sigma < 1:
            price_implied = self.S * \
                norm.cdf(self.__d1) - self.K * exp(-self.r * self.T) * \
                norm.cdf(self.__d2)
            if np.abs(price - price_implied) < 0.001:
                return sigma
            sigma += 0.001
        return np.nan

    def put_implied_volatility(
            self,
            price: np.float = np.nan,
    ) -> np.float:
        """Expected future volatiloty of underlying within time to maturity."""
        sigma = 0.001
        while sigma < 1:
            price_implied = self.K * exp(-self.r * self.T) - \
                            self.S + \
                            self.__call
            if np.abs(price - price_implied) < 0.001:
                return sigma
            sigma += 0.001
        return np.nan
