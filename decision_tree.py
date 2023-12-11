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
            self._print_recurr(self._root)

    def _print_recurr(self, current_node):
        children_line = ''

        for child in current_node.children:
            if child.is_leaf():
                print(child.class_name)
            else:
                children_line += str(child.feature_value_branch) + ',' + str(child.feature) + '       '
                print(children_line)
                self._print_recurr(child)
