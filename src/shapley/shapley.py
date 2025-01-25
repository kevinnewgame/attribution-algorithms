# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:49:05 2024

@author: ccw
"""
import pandas as pd
from itertools import chain


class UnorderdShapley:

    def __init__(self, data: pd.DataFrame):
        """df columns: journey: list, y: float"""
        self.P = frozenset(chain(*data["journey"]))

        # prepare data for simple shapley(unordered)
        df = data.copy()
        df["journey"] = df["journey"].apply(frozenset)
        df = df.groupby("journey", as_index=False)[["y"]].sum()
        df["cardinality"] = df["journey"].apply(len)
        self.df = df

    def phi(self, channel):
        df = self.df.loc[lambda df: df["journey"].apply(lambda x: channel in x)]
        return (df["y"] / df["cardinality"]).sum()

    def attribute(self):
        return {ch: self.phi(ch) for ch in self.P}


if __name__ == '__main__':
    import pandas as pd
    import json

    with open("tests/dataset/input/sample.json", "r") as f:
        journeys = json.load(f)

    df = pd.DataFrame([[j] for j in journeys], columns=["journey"])
    df["y"] = 1.0  # assume target is the conversion

    model = UnorderdShapley(df)
    model.phi(9)
    attribution = model.attribute()

    # compare to the original algorithm
    res_df = pd.DataFrame.from_dict(attribution, orient="index")
    res_df.columns = ["shapley_value"]
