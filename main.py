import pandas as pd
from id3 import *


if __name__ == '__main__':
    data = pd.read_csv('data/breast-cancer.data', sep=',', dtype=str, header=None)

    count_entropy(data)
