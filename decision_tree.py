from id3 import id3


class DecisionTree:
    def __init__(self):
        self._root = None

    def build_id3_tree(self, dataset, class_column_index):
        class_column = dataset[dataset.columns[class_column_index]]
        class_names = class_column.unique()

        num_of_columns = dataset.shape[1]
        informative_features = [i for i in range(num_of_columns)]

        class_column_index_to_remove = num_of_columns - 1 if class_column_index == -1 else class_column_index
        informative_features.remove(class_column_index_to_remove)

        self._root = id3(class_names, informative_features, dataset, class_column_index)

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
