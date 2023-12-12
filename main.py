from dataset_reader import read_dataset
from decision_tree import DecisionTree


BREST_CANCER_DATASET_PATH = 'data/breast-cancer.data'
BREST_CANCER_CLASS_INDEX = 0

AGARICUS_LEPIOTA_DATASET_PATH = 'data/agaricus-lepiota.data'
AGARICUS_LEPIOTA_CLASS_INDEX = 0

TEST_DATASET_PATH = 'data/test.data'
TEST_DATASET_CLASS_INDEX = -1


if __name__ == '__main__':
    data = read_dataset(TEST_DATASET_PATH)

    num_rows = data.shape[0]
    split_point = int(num_rows * 3 / 5)

    training_data = data[:split_point]
    testing_data = data[split_point:]

    tree = DecisionTree()
    tree.build_id3_tree(data, TEST_DATASET_CLASS_INDEX)
    tree.print()

    predictions = tree.predict(data)
    print(predictions)
