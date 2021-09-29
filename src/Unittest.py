#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 3 02:02:51 2020

Unit test
@author: Tianyi Zhang (19202673)

"""

import unittest
from StockQuotesApplication import get_size
import pandas as pd
import numpy as np


class TestClass(unittest.TestCase):

    def check_get_size(self, data, expected_return_value):
        """
        to test whether the return value of function get_size() is equal to expected_return_value
        :param data: pandas DataFrame
        :param expected_return_value: int type
        """
        return_value = get_size(data)
        self.assertEqual(return_value, expected_return_value)

    def test_size_1(self):
        df1 = pd.DataFrame(np.arange(48).reshape((6, 8)))
        self.check_get_size(df1, 4)

    def test_size_2(self):
        df2 = pd.DataFrame(np.random.randn(4, 4), columns=['a', 'b', 'c', 'd'])
        self.check_get_size(df2, 3)

    def test_size_3(self):
        df3 = pd.DataFrame(np.arange(10000).reshape((2500, 4)))
        self.check_get_size(df3, 2000)

    def test_size_4(self):
        df4 = pd.DataFrame(np.arange(361).reshape((19, 19)))
        self.check_get_size(df4, 15)


if __name__ == '__main__':
    unittest.main()
