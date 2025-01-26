# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:25:15 2024

@author: ccw
"""
import pytest
import json
import pandas as pd
from data_module.marketing_data import (
    marketing_journey_unordered_dataset,
    marketing_unordered_attribution
    )
from algorithm import shapley_unordered
from simplified_shapley_attribution_model import SimplifiedShapleyAttributionModel


@pytest.fixture(scope="function")
def journey_from_json():
    with open("data/sample.json", "r") as file:
        yield json.load(file)
        file.close()


def dict_equal(dict1, dict2):
    for k, v in dict1.items():
        assert (dict2[k] - v) < 1e-10


class TestShapley:

    def test_shapley_on_json_data(self, journey_from_json):
        journeys = journey_from_json
        # make test data as dataframe to follow API of shapley_unordered
        df = pd.DataFrame([[j] for j in journeys], columns=["journey"])
        df["converted"] = 1  # assume target is the conversion

        # shapley model
        shapley_model = shapley_unordered(df, y="converted", x="journey")
        attribution = shapley_model.attribute()

        # answer
        model_ref = SimplifiedShapleyAttributionModel()
        answer = model_ref.attribute(journeys)

        dict_equal(attribution, answer)

    def test_shapley_on_marketing_data(self):
        data = marketing_journey_unordered_dataset()
        answer = marketing_unordered_attribution()

        shapley_model = shapley_unordered(
            data, y="converted", x="marketing_channel")
        attribution = shapley_model.attribute()

        dict_equal(attribution, answer)
