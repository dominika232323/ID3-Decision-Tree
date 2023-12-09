import pandas as pd
from id3 import *


if __name__ == '__main__':
    data = pd.read_csv('data/test.data', sep=',', dtype=str, header=None)

    print(count_entropy(data))
