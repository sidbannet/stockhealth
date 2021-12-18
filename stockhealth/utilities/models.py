#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth models
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

from scipy.stats import norm
import numpy as np


# noinspection PyPep8Naming
class Greeks:
    """Class for getting greeks using Black-Scholes (European Options) model."""

    def __init__(self):
        """Instantiate the class."""
        self.__gammas = np.vectorize(
            lambda sigmas, Ks, Ss, Ts, rs, qs: norm.pdf(
                self.__f1(sigma=sigmas, S=Ss, K=Ks, T=Ts, r=rs, q=qs)
            ) / (Ss * sigmas * np.sqrt(Ts)) * np.exp(-qs * Ts)
        )
        self.__vegas = np.vectorize(
            lambda sigmas, Ks, Ss, Ts, rs, qs: (
                Ss * norm.pdf(
                    self.__f1(sigma=sigmas, S=Ss, K=Ks, T=Ts, r=rs, q=qs)
                ) * np.sqrt(Ts)
            ) * np.exp(-qs * Ts)
        )

    @staticmethod
    def __f1(sigma, S, K, T, r, q):
        return (
            np.log(S / K) +
            (r - q + sigma ** 2 / 2.0) * T
        ) / (sigma * np.sqrt(T))
    
    @staticmethod
    def __f2(sigma, S, K, T, r, q):
        return (
            np.log(S / K) +
            (r - q + sigma ** 2 / 2.0) * T
        ) / (sigma * np.sqrt(T)) - (sigma * np.sqrt(T))

    def __call_price(self, sigma, S, K, T, r, q):
        return S * norm.cdf(
            self.__f1(sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        ) - K * norm.cdf(
            self.__f2(sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        ) * np.exp(-r * T)

    def __put_price(self, sigma, S, K, T, r, q):
        return np.exp(-r * T) * K * norm.cdf(
            - self.__f2(sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        ) - S * norm.cdf(
            - self.__f1(sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        )

    def __call_sigma(self, price, S, K, T, r, q):
        sigma_ = np.linspace(start=0.001, stop=5.0, num=5000, endpoint=True)
        return sigma_[
            (
                np.abs(
                    self.__call_price(
                        sigma=sigma_, S=S, T=T, K=K, r=r, q=q
                    ) - price
                )
            ).argmin()
        ]

    def __put_sigma(self, price, S, K, T, r, q):
        sigma_ = np.linspace(start=0.001, stop=5.0, num=5000, endpoint=True)
        return sigma_[
            (
                np.abs(
                    self.__put_price(
                        sigma=sigma_, S=S, T=T, K=K, r=r, q=q,
                    ) - price
                )
            ).argmin()
        ]

    def calls(
        self,
        strikes: np.array,
        prices: np.array,
        times_to_expiry: np.array,
        interest_rate: np.float = np.nan,
        dividend_yield: np.float = 0.0,
    ) -> dict:
        """Get sigmas, and greeks for calls given stock underlying."""
        Ks_in = strikes
        Ss_in = prices
        Ts_in = times_to_expiry
        rs_in = np.full_like(strikes, interest_rate)
        qs_in = np.full_like(strikes, dividend_yield)
        sigmas_ = np.vectorize(
            lambda call_prices, Ss, Ts, Ks, rs, qs: self.__call_sigma(
                price=call_prices, S=Ss, T=Ts, K=Ks, r=rs, q=qs
            )
        )
        sigmas_in = sigmas_(
            call_prices=prices, Ss=Ss_in, Ts=Ts_in, Ks=Ks_in, rs=rs_in, qs=qs_in,
        )
        __deltas = np.vectorize(
            lambda sigmas, Ks, Ss, Ts, rs, qs: norm.cdf(
                self.__f1(
                    sigma=sigmas, S=Ss, K=Ks, T=Ts, r=rs, q=qs,
                )
            ) * np.exp(-qs * Ts) 
        )
        deltas = __deltas(
            sigmas=sigmas_in, Ks=Ks_in, Ss=Ss_in, Ts=Ts_in, rs=rs_in, qs=qs_in
        )
        gammas = self.__gammas(
            sigmas=sigmas_in, Ks=Ks_in, Ss=Ss_in, Ts=Ts_in, rs=rs_in, qs=qs_in)
        vegas = self.__vegas(sigmas=sigmas_in, Ks=Ks_in, Ss=Ss_in, Ts=Ts_in, rs=rs_in, qs=qs_in)
        __rhos = np.vectorize(
            lambda sigmas, Ks, Ss, Ts, rs, qs: (
                Ks * Ts * np.exp(-rs * Ts) * norm.cdf(
                    self.__f2(sigma=sigmas, S=Ss, K=Ks, T=Ts, r=rs, q=qs)
                )
            )
        )
        rhos = __rhos(sigmas=sigmas_in, Ks=Ks_in, Ss=Ss_in, Ts=Ts_in, rs=rs_in, qs=qs_in)
        __thetas = np.vectorize(
            lambda sigmas, Ks, Ss, Ts, rs, qs: (
                - np.exp(-qs * Ts) * (
                    Ss * norm.pdf(
                        self.__f1(sigma=sigmas, S=Ss, K=Ks, T=Ts, r=rs, q=qs)
                    ) * sigmas
                ) / (2 * np.sqrt(Ts)) -
                rs * Ks * np.exp(-rs * Ts) * norm.cdf(
                    self.__f2(sigma=sigmas, S=Ss, K=Ks, T=Ts, r=rs, q=qs)
                ) + qs * Ss * np.exp(-qs * Ts) * norm.cdf(
                    self.__f1(sigma=sigmas, S=Ss, K=Ks, T=Ts, r=rs, q=qs)
                )
            ) 
        )
        thetas = __thetas(sigmas=sigmas_in, Ks=Ks_in, Ss=Ss_in, Ts=Ts_in, rs=rs_in, qs=qs_in)
        return {
            'delta': deltas,
            'gamma': gammas,
            'rho': rhos,
            'vega': vegas,
            'theta': thetas,
            'sigma': sigmas_in,
        }

    def puts(
        self,
        strikes: np.array,
        prices: np.array,
        times_to_expiry: np.array,
        interest_rate: np.float = np.nan,
        dividend_yield: np.float = 0.0,
    ) -> dict:
        """Get sigmas, and greeks for calls given stock underlying."""
        Ks_in = strikes
        Ss_in = prices
        Ts_in = times_to_expiry
        rs_in = np.full_like(strikes, interest_rate)
        qs_in = np.full_like(strikes, dividend_yield)
        sigmas_ = np.vectorize(
            lambda put_prices, Ss, Ts, Ks, rs, qs: self.__put_sigma(
                price=put_prices, S=Ss, T=Ts, K=Ks, r=rs, q=qs
            )
        )
        sigmas_in = sigmas_(
            put_prices=prices, Ss=Ss_in, Ts=Ts_in, Ks=Ks_in, rs=rs_in, qs=qs_in,
        )
        __deltas = np.vectorize(
            lambda sigmas, Ks, Ss, Ts, rs, qs: - norm.cdf(
                -self.__f1(sigma=sigmas, S=Ss, K=Ks, T=Ts, r=rs, q=qs)
            ) * np.exp(-qs * Ts)
        )
        deltas = __deltas(
            sigmas=sigmas_in, Ks=Ks_in, Ss=Ss_in, Ts=Ts_in, rs=rs_in, qs=qs_in)
        gammas = self.__gammas(
            sigmas=sigmas_in, Ks=Ks_in, Ss=Ss_in, Ts=Ts_in, rs=rs_in, qs=qs_in)
        vegas = self.__vegas(
            sigmas=sigmas_in, Ks=Ks_in, Ss=Ss_in, Ts=Ts_in, rs=rs_in, qs=qs_in)
        __rhos = np.vectorize(
            lambda sigmas, Ks, Ss, Ts, rs, qs: (
                -Ks * Ts * np.exp(
                    -rs * Ts
                ) * norm.cdf(-self.__f2(sigma=sigmas, K=Ks, S=Ss, T=Ts, r=rs, q=qs))
            ) 
        )
        rhos = __rhos(sigmas=sigmas_in, Ks=Ks_in, Ss=Ss_in, Ts=Ts_in, rs=rs_in, qs=qs_in)
        __thetas = np.vectorize(
            lambda sigmas, Ks, Ss, Ts, rs, qs: (
                - np.exp(
                    -qs * Ts
                ) * (
                    Ss * norm.pdf(
                        self.__f1(sigma=sigmas, K=Ks, S=Ss, T=Ts, r=rs, q=qs)
                    ) * sigmas
                ) / (2 * np.sqrt(Ts)) +
                rs * Ks * np.exp(-rs * Ts) * norm.cdf(
                    -self.__f2(sigma=sigmas, S=Ss, K=Ks, T=Ts, r=rs, q=qs)
                ) -
                qs * Ss * np.exp(
                    -qs * Ts
                ) * norm.cdf(-self.__f1(sigma=sigmas, S=Ss, K=Ks, T=Ts, r=rs, q=qs))
            ) 
        )
        thetas = __thetas(sigmas=sigmas_in, Ks=Ks_in, Ss=Ss_in, Ts=Ts_in, rs=rs_in, qs=qs_in)
        return {
            'delta': deltas,
            'gamma': gammas,
            'rho': rhos,
            'vega': vegas,
            'theta': thetas,
            'sigma': sigmas_in,
        }
