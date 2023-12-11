import numpy as np
from node import Node


def id3(class_names, informative_features, dataset, class_index):
    if len(class_names) == 1:
        return Node(class_name=class_names[0])

    if len(informative_features) == 0:
        return Node(class_name=find_most_common_class(class_names, dataset))

    best_info_feature = find_max_informative_feature(informative_features, dataset)
    best_feature_column = dataset[dataset.columns[best_info_feature]]
    best_feature_values = best_feature_column.unique()

    root = Node(feature=best_info_feature)

    for value in best_feature_values:
        value_dataset = dataset.loc[dataset[best_info_feature] == value]
        new_dataset = value_dataset.drop(value_dataset.columns[best_info_feature], axis=1)

        new_informative_features = informative_features.copy()
        new_informative_features.remove(best_info_feature)

        new_class_names_column = new_dataset[new_dataset.columns[class_index]]
        new_class_names = new_class_names_column.unique()

        child = id3(new_class_names, new_informative_features, new_dataset, class_index)
        child.set_feature_value_branch(value)
        root.add_child(child)

    return root


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
