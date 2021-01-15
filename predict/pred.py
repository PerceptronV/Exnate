import requests
r = requests.get('https://raw.githubusercontent.com/PerceptronV/Exnate/master/train/models.py')
open('models.py', 'wb').write(r.content)
r = requests.get('https://raw.githubusercontent.com/PerceptronV/Exnate/master/data/dataloader.py')
open('dataloader.py', 'wb').write(r.content)

import json
import numpy as np
import tensorflow as tf
from datetime import datetime as dt
from dataloader import load_date, get_features
from preproc import pred_preprocess, interpret_pred
from models import ForecastModel

def predict(date1, date2):
  pred_df = get_features(date1, date2)[0]
  print(pred_df.iloc[-data_param['hist_len']:,:4])

  pred_data = np.asarray(
      pred_df.values,
      dtype = np.float32
  )
  pred_features, pred_residuals = pred_preprocess(pred_data, data_param, model_param)
  result = pred_model(pred_features, pred_residuals)
  print(result.shape)

  interpret_pred(result, data_param, feature_names)

  return result


def build():
  print('Building model by passing in real data...')
  pred_df = get_features(dt(2018, 11, 1), dt(2019, 1, 3))[0]

  pred_data = np.asarray(
      pred_df.values,
      dtype=np.float32
  )
  pred_features, pred_residuals = pred_preprocess(pred_data, data_param, model_param)
  result = pred_model(pred_features, pred_residuals)


subdir = 'downloads/'
data_param_fname = 'data_params.json'
model_param_fname = 'model_params.json'
feat_fname = 'feature_names.json'
weights_fname = 'weights.h5'


data_param = json.load(open(subdir + data_param_fname, 'r'))
model_param = json.load(open(subdir + model_param_fname, 'r'))
feature_names = json.load(open(subdir + feat_fname, 'r'))

pred_model = ForecastModel(model_param['input_shape'], model_param['residual_shape'],
                           model_param['output_shape'], model_param['return_sequences'],
                           model_param['rnn_units'])

# Build run
build()

# Load weights
pred_model.load_weights(subdir + weights_fname)

print('\nMain prediction: ')
_ = predict(dt(2020, 11, 1), dt(2021, 1, 3))
