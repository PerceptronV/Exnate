import requests
r = requests.get('https://raw.githubusercontent.com/PerceptronV/Exnate/master/dataloading/dataloader.py')
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
from dataloader import load_date, get_features

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.titlecolor']='green'
mpl.rcParams['axes.labelcolor']='black'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
