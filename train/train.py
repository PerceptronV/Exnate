import requests
r = requests.get('https://raw.githubusercontent.com/PerceptronV/Exnate/master/data/dataloader.py')
open('dataloader.py', 'wb').write(r.content)

import os
import json
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from models import ForecastModel


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
feature_names = json.load(open(json_path, 'r'))

np_data = np.asarray(df.iloc[:,:].values, np.float32)
indices = list(df.index)

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
