import requests
r = requests.get('https://raw.githubusercontent.com/PerceptronV/Exnate/master/train/models.py')
open('models.py', 'wb').write(r.content)
r = requests.get('https://raw.githubusercontent.com/PerceptronV/Exnate/master/data/dataloader.py')
open('dataloader.py', 'wb').write(r.content)

import numpy as np
from datetime import datetime as dt
from dataloader import load_date, get_features
from utils import pred_preprocess, interpret_pred

def predict(date1, date2, pred_model, data_param, model_param, feature_names, progress=False, interpret=True):
    pred_df = get_features(date1, date2, progress=progress)[0]

    if interpret:
        print('Data given to model: ')
        print(pred_df.iloc[-data_param['hist_len']:])

    pred_data = np.asarray(
        pred_df.values,
        dtype = np.float32
    )
    pred_features, pred_residuals = pred_preprocess(pred_data, data_param, model_param)
    result = pred_model(pred_features, pred_residuals)

    if interpret:
        interpret_pred(result, data_param, feature_names)

    return result


def build(pred_model, data_param, model_param):
    pred_df = get_features(dt(2018, 11, 1), dt(2019, 1, 3), progress=False)[0]

    pred_data = np.asarray(
          pred_df.values,
        dtype=np.float32
    )
    pred_features, pred_residuals = pred_preprocess(pred_data, data_param, model_param)
    pred_model(pred_features, pred_residuals)

    return pred_model
