import pandas as pd
from id3 import *


if __name__ == '__main__':
    data = pd.read_csv('data/test.data', sep=',', dtype=str, header=None)
    num_of_colums = len(data.columns)

    print(num_of_colums)

    print(id3(['yes'], [i for i in range(num_of_colums-1)], data))

    print(list(data[data.columns[0]]))
    print(list(data[data.columns[0]])[0])
