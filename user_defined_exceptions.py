#!/usr/bin/python3


"""
User defined exceptions
"""


__author__  = 'Zsolt Forray'
__license__ = 'MIT'
__version__ = '0.0.1'
__date__    = '05/12/2019'
__status__  = 'Development'


class InvalidTickersError(Exception):
    pass


class InvalidFeaturesError(Exception):
    pass


class InvalidParamsError(Exception):
    pass


class InvalidModelError(Exception):
    pass
