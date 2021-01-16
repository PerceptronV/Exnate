# Model training
Train your own model from a wealth of financial data

## Usage

### Prerequisites

* Tensorflow 2

* Pandas

* tqdm

* matplotlib (for interactive training)

### Training

Running `train.py` will train a model without any data visualization.

To visualize data and model architecture, run `train_interactive.py`.

Various Python files and datafiles (CSVs) will be downloaded in both processes. Several image files may also be generated if you choose to run `train_interactive.py`. 

Model checkpoints will be stored by default in the 'checkpoints/' subdirectory. You can configure this, and other default settings, by editing the `init_params()` function in either Python files.

Sample `init_params()` function:

```python
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
```
