import pandas as pd


def read_dataset(path):
    data = pd.read_csv(path, sep=',', dtype=str, header=None)
    data.sample(frac=1)

    return data
