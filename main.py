from collections import Counter
from dataset_reader import *
from decision_tree import DecisionTree


if __name__ == '__main__':
    data = read_dataset('data/test.data')

    num_rows = data.shape[0]
    split_point = int(num_rows * 3 / 5)

    training_data = data[:split_point]
    testing_data = data[split_point:]

    tree = DecisionTree()
    tree.build_id3_tree(data, -1)
    tree.print()
