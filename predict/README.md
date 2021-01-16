# Predict exchange rates

## Usage

### Prerequisites

* Tensorflow 2

* Pandas

* investpy

* tqdm

* matplotlib (for interactive training)

### Prediction

Run `main.py` fo an interactive quick start.

Note that several Python files will be downloaded in the process to build models and acquire data.

#### Custom models

To use a custom model:

1. Copy your `data_params.json`, `model_params.json`, and `weights.h5` files into the hyperparameters subdirectory, which defaults to `./downloads/`.

    * _The subdirectory path is defined in line 7 in `main.py`_

2. Follow the instructions in `get_params.py` under the `get_params()` function.

3. Run `main.py`
