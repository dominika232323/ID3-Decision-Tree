import numpy as np


def id3(Y, D, U):
    pass


def inf_gain(d, U):
    return count_entropy(U) - count_subset_entropy(d, U)


def count_entropy(U):
    I = 0

    classes = U[U.columns[-1]]
    classnames = classes.unique()

    num_of_rows = len(U)

    for name in classnames:
        class_count = classes.value_counts().get(name)
        ratio = class_count / num_of_rows

        if ratio == 0 or ratio == 1:
            class_entropy = 0
        else:
            class_entropy = -1 * ratio * np.log2(ratio)

        I += class_entropy

    return I


def count_subset_entropy(d, U):
    inf = 0

    column = U[U.columns[d]]
    column_values = column.unique()

    classes = U[U.columns[-1]]
    classnames = classes.unique()

    num_of_rows = len(U)

    for value in column_values:
        value_count = column.value_counts().get(value)

        class_count = {name: 0 for name in classnames}

        for val, cla in zip(column, classes):
            if val == value:
                class_count[cla] += 1

        subset_entropy = 0

        for name in classnames:
            ratio = class_count[name] / value_count

            if ratio == 0 or ratio == 1:
                ent = 0
            else:
                ent = -1 * ratio * np.log2(ratio)

            subset_entropy += ent

        inf += value_count / num_of_rows * subset_entropy

    return inf
