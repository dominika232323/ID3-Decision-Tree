from itertools import product
from dataset_reader import read_dataset
from decision_tree import DecisionTree


BREST_CANCER_DATASET_PATH = 'data/breast-cancer.data'
BREST_CANCER_CLASS_INDEX = 0

AGARICUS_LEPIOTA_DATASET_PATH = 'data/agaricus-lepiota.data'
AGARICUS_LEPIOTA_CLASS_INDEX = 0

TEST_DATASET_PATH = 'data/test.data'
TEST_DATASET_CLASS_INDEX = -1


def expected_vs_predicted(expected, predicted):
    expected_class_names = list(set(expected))
    predicted_class_names = list(set(predicted))
    results = {combination: 0 for combination in list(product(expected_class_names, predicted_class_names))}

    for expectation, prediction in zip(expected, predicted):
        results[(expectation, prediction)] += 1

    return results


if __name__ == '__main__':
    dataset_path = BREST_CANCER_DATASET_PATH
    dataset_class_index = BREST_CANCER_CLASS_INDEX

    data = read_dataset(dataset_path)
    data = data.sample(frac=1)

    class_column = data[dataset_class_index]
    class_names = class_column.unique()

    print(class_names)

    num_rows = data.shape[0]
    split_point = int(num_rows * 3 / 5)

    training_data = data[:split_point]
    testing_data = data[split_point:]

    tree = DecisionTree()
    tree.build_id3_tree(training_data, dataset_class_index)
    tree.print()

    expected = data[data.columns[dataset_class_index]].values.flatten().tolist()
    print(expected)

    predictions = tree.predict(testing_data)
    print(predictions)

    print(expected_vs_predicted(expected, predictions))
