#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth simulator
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

from datetime import date
import numpy as np
from stockhealth.model import _NUMBER_OF_CALENDAR_DAYS_PER_YEAR as _NCD
_NUMBER_OF_SECONDS_PER_DAY: float = 86400


def dt(now: date or float_, reference: date or float_) -> float:
    """Give the date delta."""
    if type(now) is date:
        days = (now - reference).days + \
            (now - reference).seconds / _NUMBER_OF_SECONDS_PER_DAY
        return days / _NCD
    else:
        return now - reference
