# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:58:03 2025

@author: ccw
"""
import pandas as pd


def marketing_journey_unordered_dataset():
    """user's journey should be arranged by date
    """
    split_symbol = ", "

    df = pd.read_csv(
        "data/marketing-clean.csv",
        usecols=['user_id', 'date_served', 'marketing_channel', 'converted'],
        dtype={"converted": int},
        )
    assert not df.isnull().values.any()

    df = df.drop_duplicates()

    df['date_served'] = pd.to_datetime(
        df['date_served'], format='%m/%d/%y', errors='coerce'
        )

    df = (
        df.groupby(["user_id"], as_index=False)
        .agg({"marketing_channel": lambda s: split_symbol.join(s.sort_values().unique()),
              "converted": "max"})
        .groupby(["marketing_channel"], as_index=False)[["converted"]].sum()
        )

    df["marketing_channel"] = (
        df["marketing_channel"].apply(lambda x: x.split(split_symbol))
        )

    return df


def marketing_unordered_attribution():
    return {
        'Email': 36.333333333333336,
        'Facebook': 61.833333333333336,
        'House Ads': 163.66666666666666,
        'Instagram': 57.50000000000001,
        'Push': 22.666666666666668
        }


if __name__ == '__main__':
    data = marketing_journey_unordered_dataset()
    answer = marketing_unordered_attribution()
