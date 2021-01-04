import requests
import pandas as pd
import json
from tqdm import tqdm

BASEURL = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/'
COMPACTKEY = 'CompactData/{}/{}.{}.{}?startPeriod={}&endPeriod={}'
DATAFLOW = 'Dataflow'
DATASTRUCT = 'DataStructure/{}'


def get_datasets(save=False, fname='datasets.json'):
    url = BASEURL + DATAFLOW
    r = requests.get(url).json()['Structure']['Dataflows']['Dataflow']

    datasets = {data['Name']['#text']: data['@id'] for data in r}

    if save:
        json.dump(datasets, open(fname, 'w'), indent=4, sort_keys=True)

    return datasets


def get_dataset_struct(db, save=False, fname='datastruct.json'):
    if ' ' in db:
        db = get_datasets()[0][db]

    url = BASEURL + DATASTRUCT.format(db)
    r = requests.get(url).json()['Structure']['CodeLists']['CodeList']

    codes = {code['Name']['#text']: {member['@value']: member['Description']['#text'] for member in code['Code']} for
             code in r}

    if save:
        json.dump(codes, open(fname, 'w'), indent=4, sort_keys=True)

    return codes


def get_imf(areas: list, keys: list, db='IFS', f='M', date1='', date2='', save=False, fname='imf_data.csv'):
    url = BASEURL + COMPACTKEY.format(
        db, f, '+'.join(list(areas)), '+'.join(list(keys)), date1, date2
    )

    data = requests.get(url).json()['CompactData']['DataSet']['Series']

    cols = []
    df = pd.DataFrame()
    key2des = get_dataset_struct(db)['Indicator']

    for i in tqdm(data):
        cols.append('{}: {}'.format(
            i['@REF_AREA'], key2des[i['@INDICATOR']]
        ))

        obs = pd.DataFrame()

        for o in i['Obs']:
            row = pd.DataFrame(
                [float(o['@OBS_VALUE'])], index=[o['@TIME_PERIOD']]
            )
            obs = obs.append(row)

        df = pd.concat([df, obs], axis=1)

    df.columns = cols

    if save:
        df.to_csv(fname)

    return df, cols
