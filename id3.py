import numpy as np
from leaf import Leaf


def id3(class_names, informative_features, dataset):
    if len(class_names) == 1:
        return Leaf(class_names[0])

    if len(informative_features) == 0:
        return Leaf(find_most_common_class(class_names, dataset))

    best_info_feature = find_max_informative_feature(informative_features, dataset)
    return best_info_feature


def find_most_common_class(class_names, dataset):
    most_common_class = class_names[0]

    for class_name in class_names:
        classes_column = dataset[dataset.columns[-1]]
        class_count = classes_column.value_counts().get(class_name)

        if class_count > most_common_class:
            most_common_class = class_name

    return most_common_class


def find_max_informative_feature(informative_features, dataset):
    max_inf_feature = informative_features[0]
    max_inf_gain = informative_gain(max_inf_feature, dataset)

    for d in informative_features[1:]:
        feature_gain = informative_gain(d, dataset)

        if feature_gain > max_inf_gain:
            max_inf_gain = feature_gain
            max_inf_feature = d

    return max_inf_feature


def informative_gain(inf_feature, dataset):
    return count_entropy(dataset) - count_subset_entropy(inf_feature, dataset)


def count_entropy(dataset):
    I = 0

    classes = dataset[dataset.columns[-1]]
    classnames = classes.unique()

    num_of_rows = len(dataset)

    for name in classnames:
        class_count = classes.value_counts().get(name)
        ratio = class_count / num_of_rows

        if ratio == 0 or ratio == 1:
            class_entropy = 0
        else:
            class_entropy = -1 * ratio * np.log2(ratio)

        I += class_entropy

    return I


def count_subset_entropy(inf_feature, dataset):
    inf = 0

    column = dataset[dataset.columns[inf_feature]]
    column_values = column.unique()

    classes_column = dataset[dataset.columns[-1]]
    class_names = classes_column.unique()

    num_of_rows = len(dataset)

    for value in column_values:
        value_count = column.value_counts().get(value)

        class_count = {name: 0 for name in class_names}

        for val, cla in zip(column, classes_column):
            if val == value:
                class_count[cla] += 1

        subset_entropy = 0

        for name in class_names:
            ratio = class_count[name] / value_count

            if ratio == 0 or ratio == 1:
                ent = 0
            else:
                ent = -1 * ratio * np.log2(ratio)

            subset_entropy += ent

        inf += value_count / num_of_rows * subset_entropy

    return inf
