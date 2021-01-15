import os
import requests

def download(url, fpath):
    r = requests.get(url)
    open(fpath, 'wb').write(r.content)

subdir = 'downloads/'
data_param_fname = 'data_params.json'
model_param_fname = 'model_params.json'
feat_fname = 'feature_names.json'
weights_fname = 'weights.h5'

if not os.path.exists(subdir):
    os.mkdir(subdir)

print('Downlading model params and dependencies...')

download('https://raw.githubusercontent.com/PerceptronV/Exnate/master/gen1/data_params.json',
         subdir + data_param_fname)
download('https://raw.githubusercontent.com/PerceptronV/Exnate/master/gen1/model_params.json',
         subdir + model_param_fname)
download('https://raw.githubusercontent.com/PerceptronV/Exnate/master/gen1/feature_names.json',
         subdir + feat_fname)
download('https://raw.githubusercontent.com/PerceptronV/Exnate/master/gen1/weights.h5',
         subdir + weights_fname)

print('Success')
