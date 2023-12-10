from id3 import id3


class DecisionTree:
    def __init__(self):
        self.root = None

    def build_id3_tree(self, dataset, class_column_index):
        class_column = dataset[dataset.columns[class_column_index]]
        class_names = class_column.unique()

        num_of_columns = dataset.shape[1]
        informative_features = [i for i in range(num_of_columns)]
        informative_features.remove(class_column_index)

        self.root = id3(class_names, informative_features, dataset)

    def predict(self, data):
        pass
