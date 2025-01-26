# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:49:05 2024

@author: ccw

Reference
Paper
Zhao, K., Mahboobi, S.H., Bagheri, S.R. (2018). Shapley Value Methods for Attribution Modeling in Online Advertising. arXiv preprint arXiv:1804.05327.

Python program
https://github.com/ianchute/shapley-attribution-model-zhao-naive
"""
import pandas as pd
from itertools import chain


def shapley_unordered(
        data: pd.DataFrame,
        y: str = "converted",
        x: str = "journey",
        order=False
        ):
    """df columns:
        journey: list[str], converted: int
    """
    df = data[[y, x]].rename({y: "converted", x: "journey"}, axis=1)
    df = df.astype({"converted": "float"})
    return UnorderdShapley(df)


class UnorderdShapley:

    def __init__(self, data: pd.DataFrame):
        self.P = frozenset(chain(*data["journey"]))  # touchPoints

        # prepare data for simple shapley(unordered)
        df = data.copy()
        df["journey"] = df["journey"].apply(frozenset)
        df = df.groupby("journey", as_index=False)[["converted"]].sum()
        df["cardinality"] = df["journey"].apply(len)
        self.df = df

    def phi(self, touchpoint):
        df = self.df.loc[lambda df: df["journey"].apply(lambda x: touchpoint in x)]
        return (df["converted"] / df["cardinality"]).sum()

    def attribute(self):
        return {ch: self.phi(ch) for ch in self.P}


if __name__ == '__main__':
    import pandas as pd
    import json
    from tests.test_shapley.simplified_shapley_attribution_model import (
        SimplifiedShapleyAttributionModel)

    with open("data/sample.json", "r") as f:
        journeys = json.load(f)
        f.close()

    df = pd.DataFrame([[j] for j in journeys], columns=["journey"])
    df["converted"] = 1  # assume target is the conversion

    shapley_model = shapley_unordered(df, y="converted", x="journey")
    shapley_model.phi(9)
    attribution = shapley_model.attribute()

    # validate the correctness by original source
    model_ref = SimplifiedShapleyAttributionModel()
    answer = model_ref.attribute(journeys)

    for touchpoint, value in attribution.items():
        print("Touchpoint: {:2}  new = {:6.2f}, old = {:6.2f}".format(
            touchpoint, value, answer[touchpoint]))
        assert (answer[touchpoint] - value) < 1e-10
