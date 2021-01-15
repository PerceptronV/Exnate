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
    origin = 'https://raw.githubusercontent.com/PerceptronV/Exnate/master/full_data.csv',
    fname = csv_fname, cache_subdir = sub)

json_path = tf.keras.utils.get_file(
    origin = 'https://raw.githubusercontent.com/PerceptronV/Exnate/master/feature_names.json',
    fname = json_fname, cache_subdir = sub)


