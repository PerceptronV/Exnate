import requests
import investpy as ivpy
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime as dt
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


def add_weo(base):
    r = requests.get('https://raw.githubusercontent.com/PerceptronV/Exnate/master/weo_data_oct_2020.csv')
    open('weo_dat.csv', 'wb').write(r.content)
    df = pd.read_csv('weo_dat.csv').transpose()
    os.remove('weo_dat.csv')

    cols = []
    for i in range(df.shape[1]):
        if df.loc['WEO Country Code'][i] == '112':
            cols.append('UK: {} /{}'.format(
                df.loc['Subject Descriptor'][i],
                df.loc['Units'][i]
            ))
        elif df.loc['WEO Country Code'][i] == '111':
            cols.append('US: {} /{}'.format(
                df.loc['Subject Descriptor'][i],
                df.loc['Units'][i]
            ))

    weo = df.iloc[8:-1, :-2]
    ret = pd.DataFrame()

    for i in tqdm(list(base.index)):
        base_dat = base.loc[i].transpose()

        if str(i.year) in weo.index:
            weo_dat = weo.loc[str(i.year)]

        else:
            weo_dat = pd.DataFrame([np.nan] * len(cols))

        weo_dat.columns = cols
        #print(weo_dat)
        print(weo_dat.shape)
        merge_dat = pd.concat([base_dat, weo_dat], axis=1)
        #print(merge_dat)

        ret = ret.append(merge_dat)

    return ret


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
    df = df.rename(lambda x: 'US SP500: ' + x, axis='columns')

    return df


def get_features(date1, date2, args=(
        hkd2gbp, hkd2usd, eur2gbp, ftse100, ftse250, arca, sp500
)):
    base = pd.DataFrame()
    print('Loading exchange rate, index and etf data...')
    for func in tqdm(args):
        base = pd.concat([base, func(date1, date2)], axis=1)

    print('Loading national indicator data')
    base = add_weo(base)

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


get_save(dt(1979, 1, 1), dt(1980, 1, 2), args=[hkd2gbp])
