#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth simulator
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

import matplotlib.axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import pandas as pd


# noinspection PyUnresolvedReferences
def plot(
        dataframe: pd.DataFrame,
        fig: figure.Figure = None,
        axs: axes.Axes = None,
) -> tuple:
    """Plot timeseries statistics with uncertainty bands."""
    v = dataframe.values.copy()
    v.sort(axis=1)
    df_prob_ = pd.DataFrame(data=v, index=dataframe.index).T
    df_prob_['p'] = df_prob_.index / df_prob_.index.max()
    df_prob_ = df_prob_.set_index('p')
    df_prob__ = pd.DataFrame(
        columns=df_prob_.columns,
        index=[0.00135, 0.02275, 0.15865, 0.5, 0.84135, 0.97725, 0.99865],
    )
    df_prob = pd.concat(
        [df_prob_, df_prob__],
    ).sort_index().interpolate(axis=0).T
    if fig is None or axs is None:
        fig = plt.figure('Timeseries of statistics')
        axs = fig.subplots(nrows=1, ncols=1, sharex=True, )
    axs.fill_between(
        x=df_prob.index, y1=df_prob[0.00135], y2=df_prob[0.99865],
        where=df_prob[0.99865] > df_prob[0.00135],
        facecolor='blue', alpha=0.2, interpolate=True,
    )
    axs.fill_between(
        x=df_prob.index, y1=df_prob[0.02275], y2=df_prob[0.97725],
        where=df_prob[0.97725] > df_prob[0.02275],
        facecolor='blue', alpha=0.4, interpolate=True,
    )
    axs.fill_between(
        x=df_prob.index, y1=df_prob[0.15865], y2=df_prob[0.84135],
        where=df_prob[0.84135] > df_prob[0.15865],
        facecolor='blue', alpha=0.6, interpolate=True,
    )
    df_prob[0.5].plot(ax=axs, label='median', style='-.', color='k', )
    df_prob.mean(axis=1).plot(ax=axs, label='mean', style='.', color='k')
    axs.grid(True)
    axs.legend(['99.74 %', '95.45 %', '68.27 %', 'median', 'mean'])
    axs.set_title('Confidence Interval')
    axs.set_xlabel('Time')
    fig.suptitle('Timeseries of future spot price possibility statistics')
    fig.autofmt_xdate(rotation=45)
    return fig, axs
