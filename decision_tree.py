from collections import Counter
import numpy as np
from node import Node


class DecisionTree:
    def __init__(self):
        self._root = None

        self._class_column_name = 0

    def build_id3_tree(self, dataset, class_column_name):
        self._class_column_name = class_column_name if class_column_name >= 0 else dataset.shape[1] - 1

        class_column = dataset[self._class_column_name]
        class_names = class_column.unique()

        num_of_columns = dataset.shape[1]
        informative_features = [i for i in range(num_of_columns)]
        informative_features.remove(self._class_column_name)

        self._root = self._id3(class_names, informative_features, dataset)

    def _id3(self, class_names, informative_features, dataset):
        if len(class_names) == 1:
            return Node(class_name=class_names[0])

        if len(informative_features) == 0:
            return Node(class_name=self._find_most_common_class(dataset))

        best_info_feature = self._find_max_informative_feature(informative_features, dataset)
        best_feature_column = dataset[best_info_feature]
        best_feature_values = best_feature_column.unique()

        root = Node(feature=best_info_feature)

        for value in best_feature_values:
            value_dataset = dataset[dataset[best_info_feature] == value]
            new_dataset = value_dataset.drop(columns=best_info_feature)

            new_informative_features = informative_features.copy()
            new_informative_features.remove(best_info_feature)

            new_class_names_column = new_dataset[self._class_column_name]
            new_class_names = new_class_names_column.unique()

            child = Node()

            if len(new_class_names) == 1:
                child.set_class_name(new_class_names[0])
            else:
                child = self._id3(new_class_names, new_informative_features, new_dataset)

            child.set_feature_value_branch(value)
            root.add_child(child)

        return root

    def _find_most_common_class(self, dataset):
        class_counter = Counter(dataset[self._class_column_name])
        return class_counter.most_common(1)

    def _find_max_informative_feature(self, informative_features, dataset):
        max_inf_feature = informative_features[0]
        max_inf_gain = self._informative_gain(max_inf_feature, dataset)

        for d in informative_features[1:]:
            feature_gain = self._informative_gain(d, dataset)

            if feature_gain > max_inf_gain:
                max_inf_gain = feature_gain
                max_inf_feature = d

        return max_inf_feature

    def _informative_gain(self, inf_feature, dataset):
        return self._count_entropy(dataset) - self._count_subset_entropy(inf_feature, dataset)

    def _count_entropy(self, dataset):
        class_column = dataset[self._class_column_name]
        class_counter = Counter(class_column)

        num_of_rows = len(dataset)

        entropy = self._entropy(class_counter, num_of_rows)
        return entropy

    def _count_subset_entropy(self, inf_feature, dataset):
        subset_entropy = 0

        column = dataset[inf_feature]
        value_count = Counter(column)

        classes_column = dataset[self._class_column_name]
        class_names = classes_column.unique()

        num_of_rows = len(dataset)

        for value in value_count:
            class_count = {name: 0 for name in class_names}

            for val, cla in zip(column, classes_column):
                if val == value:
                    class_count[cla] += 1

            value_entropy = self._entropy(class_count, value_count[value])
            subset_entropy += value_count[value] / num_of_rows * value_entropy

        return subset_entropy

    def _entropy(self, class_count, num_of_rows):
        value_entropy = 0

        for name in class_count:
            ratio = class_count[name] / num_of_rows

            if ratio == 0 or ratio == 1:
                ent = 0
            else:
                ent = -1 * ratio * np.log2(ratio)

            value_entropy += ent

        return value_entropy

    def predict(self, data):
        pass

    def print(self):
        if self._root is None:
            print('Empty decision tree')
        else:
            print(self._root.feature)
            self._print(self._root, 1)

    def _print(self, current_node, depth):
        for child in current_node.children:
            if child.is_leaf():
                print('\t' * depth, child.feature_value_branch, child.class_name)
            else:
                print('\t' * depth, child.feature_value_branch, child.feature)
                self._print(child, depth + 1)
