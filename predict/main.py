import json
from get_params import get_params
from models import ForecastModel
from pred import build, predict
from datetime import datetime as dt

subdir = 'downloads/'
data_param_fname = 'data_params.json'
model_param_fname = 'model_params.json'
feat_fname = 'feature_names.json'
weights_fname = 'weights.h5'

print('Getting model dependencies...')
get_params(subdir, data_param_fname, model_param_fname, feat_fname, weights_fname)
data_param = json.load(open(subdir + data_param_fname, 'r'))
model_param = json.load(open(subdir + model_param_fname, 'r'))
feature_names = json.load(open(subdir + feat_fname, 'r'))

print('Building model by passing in real data...')
pred_model = ForecastModel(model_param['input_shape'], model_param['residual_shape'],
                           model_param['output_shape'], model_param['return_sequences'],
                           model_param['rnn_units'])
pred_model = build(pred_model, data_param, model_param)

print('Loading weights...')
pred_model.load_weights(subdir + weights_fname)

while 1:
    start = dt.strptime(input('Enter start date (yyyy-mm-dd): '), '%Y-%m-%d')
    end = dt.strptime(input('Enter end date (yyyy-mm-dd): '), '%Y-%m-%d')
    _ = predict(start, end, pred_model, data_param, model_param, feature_names)
    print('\n')
