from itertools import product
from collections import Counter
from dataset_reader import read_dataset, split_data
from decision_tree import DecisionTree


BREST_CANCER_DATASET_PATH = 'data/breast-cancer.data'
BREST_CANCER_CLASS_INDEX = 0

AGARICUS_LEPIOTA_DATASET_PATH = 'data/agaricus-lepiota.data'
AGARICUS_LEPIOTA_CLASS_INDEX = 0

TEST_DATASET_PATH = 'data/test.data'
TEST_DATASET_CLASS_INDEX = 4


def expected_vs_predicted(expected, predicted):
    expected_class_names = list(set(expected))
    predicted_class_names = list(set(predicted))

    results = {}

    for expectation, prediction in zip(expected, predicted):
        if prediction is not None:
            if (expectation, prediction) in results:
                results[(expectation, prediction)] += 1
            else:
                results[(expectation, prediction)] = 1

    return results


if __name__ == '__main__':
    dataset_path = AGARICUS_LEPIOTA_DATASET_PATH
    dataset_class_index = AGARICUS_LEPIOTA_CLASS_INDEX

    data = read_dataset(dataset_path)
    data = data.sample(frac=1)

    training_data, testing_data = split_data(data, 3, 2)

    tree = DecisionTree()
    tree.build_id3_tree(training_data, dataset_class_index)

    expected = testing_data[data.columns[dataset_class_index]].values.flatten().tolist()
    predicted = tree.predict(testing_data)

    print(expected_vs_predicted(expected, predicted))
