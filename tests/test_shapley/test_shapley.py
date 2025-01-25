# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:25:15 2024

@author: ccw
"""
import pandas as pd
import json
from src.shapley import UnorderdShapley
from tests.test_shapley.make_data import get_data


class TestShapley:

    def test_simple_shapley(self):

        with open("tests/test_shapley/dataset/input/sample.json", "r") as f:
            journeys = json.load(f)

        df = pd.DataFrame([[j] for j in journeys], columns=["journey"])
        df["y"] = 1.0  # assume target is the conversion

        model = UnorderdShapley(df)
        result = model.attribute()

        # compare to the original algorithm
        result_df = pd.DataFrame.from_dict(
            result, orient="index", columns=["shapley_value"])

        result_orig = pd.read_csv(
            "tests/test_shapley/dataset/result/attribution_1.csv",
            index_col=0)

        assert abs(result_df - result_orig).max().iloc[0] < 1e-10

    def test_shapley(self):

        df = get_data("tests/test_shapley/dataset/input/marketing.csv")

        model = UnorderdShapley(df)
        result = model.attribute()

        res_df = pd.DataFrame.from_dict(
            result, orient="index", columns=["shapley_value"])

        res_orig = pd.read_csv(
            "tests/test_shapley/dataset/result/attribution_2.csv", index_col=0)

        assert abs(res_df - res_orig).max().iloc[0] < 1e-10


if __name__ == '__main__':
    tester = TestShapley()
    tester.test_shapley()
