import requests
r = requests.get('https://raw.githubusercontent.com/PerceptronV/Exnate/master/data/dataloader.py')
open('dataloader.py', 'wb').write(r.content)

import os
import json
import time
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import pandas as pd
from datetime import datetime as dt
from tqdm import tqdm
from data.dataloader import load_date, get_features
from models import ForecastModel, RNNPlex


def init_params():
    # The number of past days the model will be given data from
    HIST_LEN = 7

    # Indices of the features to be fed into the model at each time step
    FEAT = [i for i in range(data.shape[-1])] # This will feed all available indices

    # The number of days the model will need to predict into the future
    PRED_LEN = 3

    # Indices of the labels to be predicted by the model at each time step
    LAB = [i for i in range(4)]

    # Whether your RNN model will return the outputs generated at all time steps
    RET_SQN = True

    # Batch size
    BATCH_SZ = 64

    # No. of units in your rnn model
    RNN_UNITS = 1024

    # No. of epochs to train the model
    EPOCHS = 1000

    # Directory to store model checkpoints
    ckpt_dir = "checkpoints/"

    return HIST_LEN, FEAT, PRED_LEN, LAB, RET_SQN, BATCH_SZ, RNN_UNITS, EPOCHS, ckpt_dir


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.titlecolor']='green'
mpl.rcParams['axes.labelcolor']='black'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'

sub='downloads/'
csv_fname='full_data.csv'
json_fname = 'feature_names.json'

if os.path.exists(sub + csv_fname):
  os.remove(sub + csv_fname)
if os.path.exists(sub + json_fname):
  os.remove(sub + json_fname)

csv_path = tf.keras.utils.get_file(
    origin = 'https://raw.githubusercontent.com/PerceptronV/Exnate/master/data/full_data.csv',
    fname = csv_fname, cache_subdir = sub)
json_path = tf.keras.utils.get_file(
    origin = 'https://raw.githubusercontent.com/PerceptronV/Exnate/master/data/feature_names.json',
    fname = json_fname, cache_subdir = sub)

df = pd.read_csv(csv_path).fillna(0)
df.set_index('Unnamed: 0', inplace = True)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
print(df.head(df.shape[0]+1))

feature_names = json.load(open(json_path, 'r'))


def autoscale(lst, n=6):
    n = len(lst) // (n - 1)
    ret = []

    for i in range(len(lst)):
        if i % n == 0 or i == len(lst) - 1:
            ret.append(lst[i])

    return ret


def plot(dataframe, feature=3, start_date=None, end_date=None, save=False, filename=None):
    indices = list(df.index)

    if start_date is None:
        start_date = indices[0]
    if end_date is None:
        end_date = indices[-1]
    sd = load_date(start_date)
    ed = load_date(end_date)

    vals = []
    dates = []
    for i in indices:
        d = load_date(i)
        if d >= sd and d <= ed:
            vals.append(df.loc[i][feature])
            dates.append(i)

    ticks = autoscale([i for i in range(len(vals))])
    dates = autoscale(dates)
    y = 'GBP->HKD'
    t = '{} from {} to {}'.format(
        feature_names[str(feature)], dates[0], dates[-1]
    )

    plt.plot(vals)
    plt.ylabel(y)
    plt.xticks(ticks, dates)
    plt.title(t)

    if save:
        plt.savefig(filename if filename != None else '{} to {}'.format(
            dates[0], dates[-1]
        ), dpi=300)
    plt.show()

    return vals

_ = plot(df, save=True)
_ = plot(df, start_date='2000-01-20', end_date='2019-01-20', save=True)
_ = plot(df, start_date='2015-01-01', end_date='2020-01-01', save=True)
_ = plot(df, start_date='2008-01-01', end_date='2010-01-01', save=True)

np_data = np.asarray(df.iloc[:,:].values, np.float32)
print(np_data[:2])
print(np_data.shape)


def standardize(arr):
    m = arr.mean(axis=0)
    s = arr.std(axis=0)

    if not s.any() == 0:
        arr = (arr - m)
        arr = arr / s
        return arr, m, s

    arr = (arr - m) / 0.00000000001
    return arr, m, s

data, mean, std = standardize(np_data)

print('Mean and std shapes: {}{}'.format(mean.shape, std.shape))
print('\nArray of standardized data (first 2):')
print(data[:2])

assert std.any() != 0
assert (data[0]*std+mean).all() == np_data[0].all()
assert data.shape == np_data.shape


def duplicate(arr, t):
    ret = []
    for i in range(t):
        ret.append(arr)
    return np.asarray(ret, np.float32)


def window(data, hist_len, feat, pred_len, lab, ret_sqn, overview=True):
    if overview:
        print('The model will learn to predict\n{}\nfor {} coming days ' \
              'based on \n{}\nfrom the past {} days of data.\n'.format(
            [feature_names[str(i)] for i in lab], pred_len,
            [feature_names[str(i)] for i in feat], hist_len
        ))

    seg_len = hist_len + pred_len

    feat_len = hist_len
    feat_dim = len(feat)

    lab_len = 1
    if ret_sqn:
        lab_len += feat_len - 1

    features_dataset = []
    residuals_dataset = []
    labels_dataset = []

    for i in range(len(data) - seg_len + 1):
        seg = data[i:i + seg_len]

        features = []
        residuals = []
        for f in range(feat_len):
            features.append([seg[f][j] for j in feat])
            if f >= feat_len - lab_len:
                residuals.append(duplicate([seg[f][j] for j in lab], pred_len))
        features_dataset.append(features)
        residuals_dataset.append(residuals)

        labels = []
        for f in range(seg_len - pred_len - lab_len + 1, seg_len - pred_len + 1):
            local_lab = []
            for j in range(f, f + pred_len):
                local_lab.append([seg[j][k] for k in lab])
            labels.append(local_lab)
        labels_dataset.append(labels)

    assert len(features_dataset) == len(labels_dataset) and \
           len(features_dataset) == len(residuals_dataset)

    print('Converting to tf.data.Dataset...')
    dataset = tf.data.Dataset.from_tensor_slices(
        (features_dataset, residuals_dataset, labels_dataset)
    )

    print('Shuffling dataset...')
    dataset.shuffle(1000000, seed=13)

    print('Success')

    return dataset, len(features_dataset), (feat_len, feat_dim), \
           (lab_len, pred_len, len(lab)), (lab_len, pred_len, len(lab))

HIST_LEN, FEAT, PRED_LEN, LAB, RET_SQN, BATCH_SZ, RNN_UNITS, EPOCHS, ckpt_dir = init_params()

dataset, dataset_size, feature_shape, res_shape, label_shape = window(
    data, HIST_LEN, FEAT, PRED_LEN, LAB, RET_SQN
)

for i in dataset.take(1):
    sample_features = i[0]
    sample_residuals = i[1]
    sample_labels = i[2]
    print('Sample features {}:\n{}\n\n' \
          'Sample residuals {}:\n{}\n\n' \
          'Sample labels {}:\n{}\n\n'.format(
        sample_features.shape, sample_features,
        sample_residuals.shape, sample_residuals,
        sample_labels.shape, sample_labels))

    assert i[0].shape == feature_shape and \
           i[1].shape == res_shape and \
           i[2].shape == label_shape

ratio = 9/10
train_no = int(ratio * dataset_size)
test_no = dataset_size - train_no
print('Number of training samples: {}'.format(train_no))
print('Number of testing samples: {}'.format(test_no))

train_dataset = dataset.take(train_no)
test_dataset = dataset.skip(train_no)

train_dataset = train_dataset.batch(BATCH_SZ)
test_dataset = test_dataset.batch(BATCH_SZ)

json.dump({
    'mean': mean.tolist(),
    'std': std.tolist(),
    'hist_len': HIST_LEN,
    'features': FEAT,
    'pred_len': PRED_LEN,
    'labels': LAB
}, open('data_params.json', 'w'))

sample_model = ForecastModel(feature_shape, res_shape, label_shape, RET_SQN, RNN_UNITS)

sample_pred = sample_model(tf.expand_dims(sample_features, 0),
                           tf.expand_dims(sample_residuals, 0))

print('Sample prediction of shape {}:\n{}'.format(
    sample_pred.shape, sample_pred
))

sample_model.summary()

model = ForecastModel(feature_shape, res_shape, label_shape, RET_SQN, RNN_UNITS)

json.dump({
  'input_shape': feature_shape,
  'residual_shape': res_shape,
  'output_shape': label_shape,
  'return_sequences': RET_SQN,
  'rnn_units': RNN_UNITS
}, open('model_params.json', 'w'))

loss_func = tf.keras.losses.MeanSquaredError()
metric = tf.keras.metrics.MeanAbsoluteError()
test_metric = tf.keras.metrics.MeanAbsoluteError()
optim = tf.keras.optimizers.Adam(1e-5)


@tf.function
def train_step(features, res, labels):
    with tf.GradientTape() as tape:
        logits = model(features, res)
        loss = loss_func(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    metric.update_state(labels, logits)

    return loss


def checkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir


def getdir(ckpt_dir, epoch, batch=None):
    if batch is None:
        return checkdir(os.path.join(ckpt_dir, 'Epc {}'.format(epoch)))

    return checkdir(os.path.join(ckpt_dir, 'Epc {}'.format(epoch), 'Btc {}'.format(batch)))


def epoch_log(line, ckpt_dir, epoch):
    p = os.path.join(getdir(ckpt_dir, epoch), 'epoch_log.txt')

    if os.path.exists(p):
        b = open(p, 'r').read()
    else:
        b = ''

    open(p, 'w').write(b + '\n' + line)


def log(l, ckpt_dir):
    print(l)

    p = os.path.join(ckpt_dir, 'main_log.txt')

    if os.path.exists(p):
        b = open(p, 'r').read()
    else:
        b = ''

    open(p, 'w').write(b + '\n' + l)


def save_ckpt(ckpt_dir, epoch):
    p = os.path.join(getdir(ckpt_dir, epoch), 'weights.h5')
    model.save_weights(p)


def train(dataset, test_dataset, epochs, ckpt_dir):
    checkdir(ckpt_dir)

    for epoch in range(epochs):
        l = 'Epoch {}/{}'.format(epoch + 1, epochs)
        log(l, ckpt_dir)
        epoch_log(l, ckpt_dir, epoch + 1)
        avg_loss = []
        start_time = time.time()

        for batch, (features, res, labels) in enumerate(train_dataset):
            step_start_time = time.time()

            loss = train_step(features, res, labels)
            epoch_log('Batch {} - loss: {}, acc: {}, step time: {}s, elapsed time: {}s'.format(
                batch, loss, metric.result(),
                time.time() - step_start_time, time.time() - start_time
            ), ckpt_dir, epoch + 1)
            avg_loss.append(loss)

            if batch % 40 == 0:
                log('>\tBatch {}   \tloss: {}, acc: {},\n\t\t\tstep time: {}s, elapsed time: {}s'.format(
                    batch, loss, metric.result(),
                    time.time() - step_start_time, time.time() - start_time
                ), ckpt_dir)

        for features_test, res_test, labels_test in test_dataset:
            logits_test = model(features_test, res_test)
            test_metric.update_state(labels_test, logits_test)

        test_metric_result = test_metric.result()
        test_metric.reset_states()

        l = 'Epoch {} completed in {}s.\nAverage loss: {}, acc: {}, test acc: {}.\n'.format(
            epoch + 1, time.time() - start_time, np.mean(avg_loss), metric.result(), test_metric_result
        )
        log(l, ckpt_dir)
        epoch_log(l, ckpt_dir, epoch + 1)
        save_ckpt(ckpt_dir, epoch + 1)

        metric.reset_states()


print('\nTraining...')
train(train_dataset, test_dataset, epochs = EPOCHS, ckpt_dir = ckpt_dir)
