# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:25:15 2024

@author: ccw
"""
import pandas as pd
import numpy as np
import pytest
from algorithm.spm import SimpleProbabilisticModel


class TestSpm:

    # def __init__(self):
    #     data = self.data = pd.read_pickle(
    #         "tests/test_spm/dataset/a_data.pkl")
    #     self.tp = "tp0_tp1"  # test touchpoint

    #     self.y = data["conversion"]
    #     self.w = data["n"]
    #     self.X = data.drop(["conversion", "n"], axis=1)
    #     self.model = SimpleProbabilisticModel(self.y, self.X, self.w)
    #     self.model.train()

    def test_first_prob(self):
        x = self.data[self.tp]
        n_positive = self.y.mul(x).mul(self.w).sum()
        n_negative = (1 - self.y).mul(x).mul(self.w).sum()
        prob = n_positive / (n_positive + n_negative)
        assert prob == self.model.prob_pairs[self.tp][self.tp]

    def test_n_pair(self):

        def test(ind):
            n_pair = self.model._cal_n_pair(ind, x)  # from program
            # by hand
            ans = (
                self.X.mul(x, axis=0)
                .mul(ind, axis=0)
                .mul(self.w, axis=0)
                .sum())
            assert ans.equals(n_pair)

        x = self.X[self.tp]
        test(self.y)  # positive
        test(1 - self.y)  # negative

    def test_n_pairs(self):

        def make_n_pairs(ind):
            return pd.concat(
                [self.model._cal_n_pair(ind, self.X[tp])
                 for tp in self.X.columns],
                axis=1)

        def test(ind):
            ans = make_n_pairs(ind)
            cal = self.model._cal_n_pairs(ind)
            cal, ans = cal.align(ans)
            assert cal.equals(ans)

        test(self.y)  # positive
        test(1 - self.y)  # negative

    def test_prob_pairs(self):
        n_positive = self.model._cal_n_pairs(self.y)
        n_negative = self.model._cal_n_pairs(1 - self.y)
        second_prob = n_positive / (n_positive + n_negative)
        assert second_prob.equals(self.model.prob_pairs)

    def test_prob_result_c(self):
        ppairs = self.model.prob_pairs
        second_order = ppairs[self.tp].dropna()
        first_order = pd.Series(np.diag(ppairs), index=ppairs.columns)
        first_order = first_order.loc[second_order.index]
        n = len(second_order) - 1

        bracket = (
            second_order.drop(self.tp)
            - first_order[self.tp]
            - first_order.drop(self.tp)
            )

        ans = first_order[self.tp] + 1 / (2 * n) * bracket.sum()
        assert abs(self.model.c(self.tp) - ans) < 1e-8

    def test_many_tps(self):
        _tp = self.tp
        for tp in self.X.columns:
            self.tp = tp
            self.test_prob_result_c()
        self.tp = _tp


if __name__ == '__main__':
    tester = TestSpm()
    tester.test_first_prob()
    tester.test_n_pair()
    tester.test_n_pairs()
    tester.test_prob_pairs()
    tester.test_prob_result_c()
    tester.test_many_tps()
