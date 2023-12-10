import pandas as pd
from id3 import *


if __name__ == '__main__':
    data = pd.read_csv('data/test.data', sep=',', dtype=str, header=None)
    data.sample(frac=1)

    num_rows = data.shape[0]
    split_point = int(num_rows * 3 / 5)

    training_data = data[:split_point]
    testing_data = data[split_point:]
