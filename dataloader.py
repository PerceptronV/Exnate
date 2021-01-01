import requests
import investpy as ivpy
import os
import locale
import json
import pandas as pd
from datetime import datetime as dt
from bs4 import BeautifulSoup
from tqdm import tqdm


def create_indices(df):
    new_indices = []

    for i in df.index:
        new_indices.append(save_date(i))
    df = df.rename(save_date, axis='index')

    return df


def load_date(idx):
    return dt.strptime(idx, '%Y-%m-%d').date()


def save_date(date):
    return date.strftime('%Y-%m-%d')


def weo(date1, date2):
    r = requests.get('https://github.com/PerceptronV/Exnate/blob/master/weo_data_oct_2020.xlsx?raw=true')
    open('weo_dat.xlsx', 'wb').write(r.content)
    df = pd.read_excel('weo_dat.xlsx')
    os.remove('weo_dat.xlsx')

    print(df.head())


def hkd2gbp(date1, date2):
    df = ivpy.get_currency_cross_historical_data(currency_cross='GBP/HKD',
                                                 from_date=date1.strftime('%d/%m/%Y'),
                                                 to_date=date2.strftime('%d/%m/%Y'))
    df = df.iloc[:, :4]
    df = df.rename(lambda x: 'HKD->GBP: ' + x, axis='columns')

    return df


def hkd2usd(date1, date2):
    df = ivpy.get_currency_cross_historical_data(currency_cross='USD/HKD',
                                                 from_date=date1.strftime('%d/%m/%Y'),
                                                 to_date=date2.strftime('%d/%m/%Y'))
    df = df.iloc[:, :4]
    df = df.rename(lambda x: 'HKD->USD: ' + x, axis='columns')

    return df


def eur2gbp(date1, date2):
    df = ivpy.get_currency_cross_historical_data(currency_cross='GBP/EUR',
                                                 from_date=date1.strftime('%d/%m/%Y'),
                                                 to_date=date2.strftime('%d/%m/%Y'))
    df = df.iloc[:, :4]
    df = df.rename(lambda x: 'EUR->GBP: ' + x, axis='columns')

    return df


def ftse100(date1, date2):
    df = ivpy.get_index_historical_data(index='FTSE 100',
                                        country='united kingdom',
                                        from_date=date1.strftime('%d/%m/%Y'),
                                        to_date=date2.strftime('%d/%m/%Y'))
    df = df.iloc[:, :4]
    df = df.rename(lambda x: 'UK FTSE 100: ' + x, axis='columns')

    return df


def ftse250(date1, date2):
    df = ivpy.get_index_historical_data(index='FTSE 250',
                                        country='united kingdom',
                                        from_date=date1.strftime('%d/%m/%Y'),
                                        to_date=date2.strftime('%d/%m/%Y'))
    df = df.iloc[:, :4]
    df = df.rename(lambda x: 'UK FTSE 250: ' + x, axis='columns')

    return df


def arca(date1, date2):
    df = ivpy.get_index_historical_data(index='ARCA Major Markets',
                                        country='united states',
                                        from_date=date1.strftime('%d/%m/%Y'),
                                        to_date=date2.strftime('%d/%m/%Y'))
    df = df.iloc[:, :4]
    df = df.rename(lambda x: 'US ARCA: ' + x, axis='columns')

    return df


def sp500(date1, date2):
    df = ivpy.get_index_historical_data(index='S&P 500',
                                        country='united states',
                                        from_date=date1.strftime('%d/%m/%Y'),
                                        to_date=date2.strftime('%d/%m/%Y'))
    df = df.iloc[:, :4]
    df = df.rename(lambda x: 'US ARCA: ' + x, axis='columns')

    return df


def get_features(date1, date2, args=(
        hkd2gbp, hkd2usd, eur2gbp, ftse100, ftse250, arca, sp500
)):
    base = pd.DataFrame()
    for func in tqdm(args):
        base = pd.concat([base, func(date1, date2)], axis=1)
    base = create_indices(base)
    base = base.fillna(0)

    features_dict = {e: i for e, i in enumerate(base.columns)}

    return base, features_dict


def get_save(date1, date2, args=None, csv_path='full_data.csv', json_path='feature_names.json'):
    if args is None:
        data, feats = get_features(date1, date2)
    else:
        data, feats = get_features(date1, date2, args)

    data.to_csv(csv_path)
    json.dump(feats, open(json_path, 'w'))


# data, feats = get_features(dt(2000, 1, 1), dt(2020, 12, 31), args=[])
# data, feats = get_features(dt(1900, 1, 1), dt(2020, 12, 31))
# print(data)
# print(feats)


weo(dt(2000, 1, 1), dt(2020, 12, 3))
