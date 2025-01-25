# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:45:37 2024

@author: ccw
"""
import pandas as pd
import numpy as np


class SimpleProbabilisticModel:

    def __init__(
            self, y: pd.Series,
            X: pd.DataFrame,
            weight: pd.Series = None):
        self.y = y.copy()
        self.X = X.copy()
        if weight is not None:
            self.w = weight.copy()
        else:
            self.w = pd.Series(np.ones_like(self.y), index=self.y.index)
        self.N = self.X.shape[1]  # total number of channels/touchpoints

    def _cal_n_pair(self, indicator: pd.Series, x: pd.Series) -> pd.Series:
        """indicator: y (positive) OR 1 - y (negative)"""
        n_xix = (
            self.X.mul(x, axis=0)
            .mul(indicator, axis=0)
            .mul(self.w, axis=0)
            .sum())
        n_xix.name = x.name
        return n_xix

    def _cal_n_pairs(self, indicator):
        """indicator: y (positive) OR 1 - y (negative)"""
        res = pd.concat(
            [self._cal_n_pair(indicator, self.X.iloc[:, i])
             for i in range(self.N)],
            axis=1)
        res.sort_index(axis=0, inplace=True)
        res.sort_index(axis=1, inplace=True)
        return res

    def train(self):
        n_p_pairs = self._cal_n_pairs(self.y)
        n_n_pairs = self._cal_n_pairs(1 - self.y)
        prob_pairs = n_p_pairs / (n_p_pairs + n_n_pairs)
        assert prob_pairs.T.equals(prob_pairs)  # is symmetric
        self.prob_pairs = prob_pairs

    def c(self, tp: str):
        # only keep tp based on second order exist (intersection)
        second_order = self.prob_pairs[tp].loc[lambda s: s.notnull()]

        first_order = pd.Series(np.diag(self.prob_pairs),
                                index=self.prob_pairs.index)
        first_order = first_order.loc[second_order.index]

        second_order_effect = (
            second_order.drop(tp)
            - first_order[tp]
            - first_order.drop(tp)
            )

        Njni = len(second_order_effect)
        second_sum = second_order_effect.sum() / (2 * Njni) if Njni > 0 else 0
        res = first_order[tp] + second_sum
        return res, second_order_effect

    def predict(self):
        tps = self.X.columns
        return pd.Series([self.c(tp)[0] for tp in tps], index=tps)

    def get_second_order_effect(self):
        tps = self.X.columns
        res = pd.concat(
            [pd.Series(self.c(tp)[1], name=tp) for tp in tps],
            axis=1)
        return res.reindex(tps, axis=0)


if __name__ == '__main__':
    from src.negative_sample.sample_journey import SampleJourney

    # price_tag = "平日天猫41-100"
    price_tag = "平日天猫21-40"

    sample = SampleJourney("dataset/pickle/tianmao_usual.pkl", price_tag)
    # data = sample.make_model_data()
    data = sample.make_more_model_data()

    # data = pd.read_pickle(
    #     "tests/test_spm/dataset/tianmao_usual21_40_model_data.pkl")

    y = data["conversion"]
    weight = data["n"]
    values = data.drop(["conversion", "n"], axis=1)

    model = SimpleProbabilisticModel(y, values, weight)
    model._cal_n_pair(y, values["搜索_搜索结果"])

    model.train()
    # model.c("会场_others")
    model.c("游戏类_浏览得奖励")
    res = model.predict()
    res.name = price_tag

    second_order_effect = model.get_second_order_effect()
