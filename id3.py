def id3(Y, D, U):
    pass


def inf_gain(d, U):
    return count_entropy(U) - count_subset_entropy(d, U)


def count_entropy(U):
    classes = U[U.columns[-1]]
    classnames = classes.unique()
    print(classnames)


def count_subset_entropy(d, U):
    pass
