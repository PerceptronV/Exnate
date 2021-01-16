import os
import requests

def download(url, fpath):
    r = requests.get(url)
    open(fpath, 'wb').write(r.content)


def get_params(subdir, data_param_fname, model_param_fname, feat_fname, weights_fname):
    if not os.path.exists(subdir):
        os.mkdir(subdir)

    download('https://raw.githubusercontent.com/PerceptronV/Exnate/master/gen1/data_params.json',
             subdir + data_param_fname)
    download('https://raw.githubusercontent.com/PerceptronV/Exnate/master/gen1/model_params.json',
             subdir + model_param_fname)
    download('https://raw.githubusercontent.com/PerceptronV/Exnate/master/gen1/feature_names.json',
             subdir + feat_fname)
    download('https://raw.githubusercontent.com/PerceptronV/Exnate/master/gen1/last_weights.h5',
             subdir + weights_fname)
