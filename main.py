import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv('data/breast-cancer.data', sep=',', dtype=str, header=None)
    classes = data[data.columns[0]]
    classnames = classes.unique()

    # print(data)
    # print()
    # print(classes)
    # print()
    # print(classnames)
    # print()
    c = data[0]
    print(c[2])
