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
        class_entropy = -1 * ratio * np.log2(ratio)
        I += class_entropy

    return I


def count_subset_entropy(d, U):
    pass
