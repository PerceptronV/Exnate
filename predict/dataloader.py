import requests
import investpy as ivpy
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime as dt
from tqdm import tqdm
import warnings

if not os.path.exists('imfloader.py'):
    r = requests.get('https://raw.githubusercontent.com/PerceptronV/Exnate/master/data/imfloader.py')
    open('imfloader.py', 'wb').write(r.content)
from imfloader import get_imf


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


def pad(i):
    if i < 10:
        return '0' + str(i)
    return str(i)


def last_month(s):
    [yr, month] = s.split('-')
    month = int(month)
    if month == 1:
        return '-'.join([str(int(yr) - 1), '12'])
    return '-'.join([yr, pad(month - 1)])


def add_imf_api(base, areas, indicators, progress):
    try:
        imf, cols = get_imf(areas, indicators, progress=progress)
        aug = pd.DataFrame()
        indices = list(base.index)

        if progress:
            iter = tqdm(range(base.shape[0]))
        else:
            iter = range(base.shape[0])

        for i in iter:
            if last_month(indices[i].strftime('%Y-%m')) in imf.index:
                imf_dat = imf.loc[last_month(indices[i].strftime('%Y-%m'))]
                imf_dat = imf_dat.rename(index=indices[i])

            else:
                imf_dat = pd.DataFrame({i: np.nan for i in cols}, index=[indices[i]])
                imf_dat.columns = cols

            aug = aug.append(imf_dat)

        return pd.concat([base, aug], axis=1)

    except:
        warnings.warn('Error in data collection')
        return base


def add_imf_legacy(base, progress):
    r = requests.get('https://raw.githubusercontent.com/PerceptronV/Exnate/master/data/weo_data_oct_2020.csv')
    open('weo_dat.csv', 'wb').write(r.content)
    df = pd.read_csv('weo_dat.csv').transpose()
    os.remove('weo_dat.csv')

    cols = []
    for i in range(df.shape[1]):
        if df.loc['WEO Country Code'][i] == '112':
            cols.append('UK: {} /{} fr last yr'.format(
                df.loc['Subject Descriptor'][i],
                df.loc['Units'][i]
            ))
        elif df.loc['WEO Country Code'][i] == '111':
            cols.append('US: {} /{} fr last yr'.format(
                df.loc['Subject Descriptor'][i],
                df.loc['Units'][i]
            ))

    weo = df.iloc[8:-1, :-2]
    weo.columns = cols
    aug = pd.DataFrame()
    indices = list(base.index)

    if progress:
        iter = tqdm(range(base.shape[0]))
    else:
        iter = range(base.shape[0])

    for i in iter:
        if '{}'.format(indices[i].year - 1) in weo.index:
            weo_dat = weo.loc['{}'.format(indices[i].year - 1)]
            weo_dat = weo_dat.rename(index=indices[i])

        else:
            weo_dat = pd.DataFrame({i:np.nan for i in cols}, index=[indices[i]])
            weo_dat.columns = cols

        aug = aug.append(weo_dat)

    return pd.concat([base, aug], axis=1)


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


def nasdaq(date1, date2):
    df = ivpy.get_index_historical_data(index='nasdaq',
                                        country='united states',
                                        from_date=date1.strftime('%d/%m/%Y'),
                                        to_date=date2.strftime('%d/%m/%Y'))
    df = df.iloc[:, :4]
    df = df.rename(lambda x: 'US SP500: ' + x, axis='columns')

    return df


def chunks(lst, n):
    lst = list(lst)
    ret = []
    for i in range(0, len(lst), n):
        ret.append(lst[i:i + n])

    return ret


def get_features(date1, date2, args=(
        hkd2gbp, hkd2usd, eur2gbp, ftse100, ftse250, sp500, nasdaq
), imf_areas=('HK', 'GB', 'US'), imf_indicators=(
    'AIP_SA_IX', 'AOMPC_IX', 'NGDP_SA_XDC', 'NC_GDP_PT', 'NFI_SA_XDC', 'NGDPNPI_SA_XDC', 'NX_SA_XDC', 'NXS_SA_XDC', 'NSDGDP_R_CH_SA_XDC', 'NYGDP_XDC',
    'ARS_IX', 'ENEER_IX', 'NYFC_XDC', 'NYG_SA_XDC', 'NYP_XDC', 'BFDA_BP6_USD', 'BFOAE_BP6_USD', 'BFPAE_BP6_USD', 'BFPLXF_BP6_USD', 'FISR_PA',
    'FITB_IX', 'FMVB_IX', 'NNL_SA_XDC', 'NSG_XDC', 'NYG_SA_XDC', 'PCPI_PC_PP_PT', 'NM_XDC', 'LUR_PC_PP_PT', 'LUR_PT', 'LP_PE_NUM',
    'FPE_IX', 'LE_IX', 'BCG_GRTI_G01_CA_XDC', 'FIDR_ON_PA', 'BCG_GRTGS_G01_XDC', 'BCG_GX_G01_XDC', 'BCG_GXOB_G01_XDC', 'BFDAE_BP6_USD', 'BCG_GXCBG_G01_XDC', 'BCG_GXOBP_G01_XDC',
    'LWR_IX', 'NGDP_D_SA_IX', 'FMD_SA_USD', 'GG_GALM_G01_XDC', '26N___XDC', 'NM_SA_XDC', 'TMG_CIF_PC_PP_PT',
    'NGDP_R_K_IX', 'PPPIFG_IX', 'TMG_D_CIF_IX', 'PMP_IX', 'PCPI_IX', 'PPPI_IX', 'PXP_IX',
), beta=False, progress=True):
    base = pd.DataFrame()

    if progress:
        print('Loading exchange rate and stock market data...')
        iter = tqdm(args)
    else:
        iter = args

    for func in iter:
        base = pd.concat([base, func(date1, date2)], axis=1)

    if beta:
        if progress:
            print('Loading IMF International Financial Statistics data...')
            iter = tqdm(chunks(imf_indicators, 10))
        else:
            iter = chunks(imf_indicators, 10)

        for ind in iter:
            base = add_imf_api(base, imf_areas, ind, progress=progress)

    if progress:
        print('Loading IMF World Economic Outlook data...')

    base = add_imf_legacy(base, progress=progress)

    base = create_indices(base)
    base = base.fillna(0)

    features_dict = {int(e): i for e, i in enumerate(base.columns)}

    return base, features_dict


def get_save(date1, date2, args=None, csv_path='full_data.csv', json_path='feature_names.json', beta=False):
    if args is None:
        data, feats = get_features(date1, date2, beta=beta)
    else:
        data, feats = get_features(date1, date2, args, beta=beta)

    data.to_csv(csv_path)
    json.dump(feats, open(json_path, 'w'))
