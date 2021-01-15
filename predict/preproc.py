import requests

import numpy as np
import pandas as pd
from dataloader import load_date, get_features


def pred_standardize(arr, data_param):
    m = np.asarray(data_param['mean'])
    s = np.asarray(data_param['std'])

    arr = (arr - m)
    arr = arr / s
    return arr


def pred_duplicate(arr, t):
    ret = []
    for i in range(t):
        ret.append(arr)
    return np.asarray(ret, np.float32)


def get_relevant(arr, data_param):
    return [arr[i] for i in data_param['labels']]


def pred_preprocess(data, data_param, model_param):
    feat_len = data_param['hist_len']
    feat = data_param['features']
    pred_len = data_param['pred_len']
    lab = data_param['labels']
    ret_sqn = model_param['return_sequences']

    lab_len = 1
    if ret_sqn:
        lab_len += feat_len - 1

    seg = pred_standardize(data, data_param)[-feat_len:]

    features = []
    residuals = []
    for f in range(feat_len):
        features.append([seg[f][j] for j in feat])
        if f >= feat_len - lab_len:
            residuals.append(pred_duplicate([seg[f][j] for j in lab], pred_len))

    return np.expand_dims(features, 0), np.expand_dims(residuals, 0)


def interpret_pred(pred, data_param, feature_names):
    header_template = '\n{} day(s) into the future:'
    member_template = '{}: {}'

    for e, day in enumerate(pred[0][-1].numpy()):
        print(header_template.format(e + 1))

        proc_day = day * get_relevant(data_param['std'], data_param) \
                   + get_relevant(data_param['mean'], data_param)

        for e2, m in enumerate(proc_day):
            print(member_template.format(
                feature_names[str(data_param['labels'][e2])], m
            ))
