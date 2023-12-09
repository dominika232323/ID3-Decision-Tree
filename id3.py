def id3(Y, D, U):
    pass


def inf_gain(d, U):
    return count_entropy(U) - count_subset_entropy(d, U)


def count_entropy(U):
    I = 0

    classes = U[U.columns[-1]]
    classnames = classes.unique()

    for name in classnames:
        class_count = classes.value_counts().get(name)
        print(name, class_count)


def count_subset_entropy(d, U):
    pass
