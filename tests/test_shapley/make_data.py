# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:50:53 2024

@author: ccw
"""
import pandas as pd


def get_data(inpath):

    data_raw = pd.read_csv(inpath)
    # extracting the needed field
    columns = ['user_id', 'date_served', 'marketing_channel', 'converted']
    data = data_raw[columns].copy()

    # dropping null values
    data.dropna(axis=0, inplace=True)
    # relabel conversion to 1/0
    data['converted'] = data['converted'].astype('int')
    # converting date_served into date format
    data['date_served'] = pd.to_datetime(
        data['date_served'], format='%m/%d/%y', errors='coerce')

    # create a channel mix conversion table
    # first level - sort
    data_lvl1 = (
        data
        .filter(['user_id', 'marketing_channel', 'converted'])
        .sort_values(by=['user_id', 'marketing_channel']))

    # second level - groupby userid, concat distinct marketing channel and label if any conversion took place with this channel mix
    data_lvl2 = (
        data_lvl1
        .groupby(['user_id'], as_index=False)
        .agg({'marketing_channel': lambda x: ','.join(map(str, x.unique())),
              'converted': max}))
    data_lvl2.rename(columns={'marketing_channel': 'marketing_channel_subset'},
                     inplace=True)

    # third level - summing up the conversion which took place for each channel mix
    data_lvl3 = (
        data_lvl2
        .groupby(['marketing_channel_subset'], as_index=False)[["converted"]]
        .sum())

    # data for my class
    df = data_lvl3
    df["journey"] = df["marketing_channel_subset"].apply(lambda x: x.split(','))
    df.rename(columns={"converted": "y"}, inplace=True)
    return df[["journey", "y"]]


if __name__ == '__main__':
    df = get_data()
