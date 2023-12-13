from tabulate import tabulate
from dataset_reader import read_dataset, split_data
from decision_tree import DecisionTree


BREST_CANCER_DATASET_PATH = 'data/breast-cancer.data'
BREST_CANCER_CLASS_INDEX = 0

AGARICUS_LEPIOTA_DATASET_PATH = 'data/agaricus-lepiota.data'
AGARICUS_LEPIOTA_CLASS_INDEX = 0

TEST_DATASET_PATH = 'data/test.data'
TEST_DATASET_CLASS_INDEX = 4


def expected_vs_predicted(expected, predicted):
    results = {}

    for expectation, prediction in zip(expected, predicted):
        if prediction is not None:
            if (expectation, prediction) in results:
                results[(expectation, prediction)] += 1
            else:
                results[(expectation, prediction)] = 1

    return results


def review_results(expected_vs_predicted_result):
    correct_result = 0
    incorrect_result = 0

    for result in expected_vs_predicted_result:
        if result[0] == result[1]:
            correct_result += expected_vs_predicted_result[result]
        else:
            incorrect_result += expected_vs_predicted_result[result]

    return correct_result / (correct_result + incorrect_result), correct_result, incorrect_result


def print_table(table):
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


def print_result(expected_vs_predicted_result, expected):
    expected_classes = list(set(expected))
    result_to_print = [['Oczekiwane / przewidziane']]

    for class_name in expected_classes:
        result_to_print[0].append(class_name)
        result_to_print.append([class_name] + [0] * len(expected_classes))

    for result in expected_vs_predicted_result:
        for index, row in enumerate(result_to_print):
            if result[0] == row[0]:
                row_index = index

        for index, col in enumerate(result_to_print[0]):
            if result[1] == col:
                column_index = index

        result_to_print[row_index][column_index] = expected_vs_predicted_result[result]

    print_table(result_to_print)


if __name__ == '__main__':
    dataset_path = AGARICUS_LEPIOTA_DATASET_PATH
    dataset_class_index = AGARICUS_LEPIOTA_CLASS_INDEX

    reviewed_results_info = [['Liczba wykonanych przewidzeń', 'Liczba poprawnych przewidzeń', 'Liczba niepoprawnych przewidzeń', 'Dokładność przewidzeń']]

    for i in range(20):
        data = read_dataset(dataset_path)
        data = data.sample(frac=1)

        training_data, testing_data = split_data(data, 3, 2)

        tree = DecisionTree()
        tree.build_id3_tree(training_data, dataset_class_index)

        expected = testing_data[data.columns[dataset_class_index]].values.flatten().tolist()
        predicted = tree.predict(testing_data)

        results = expected_vs_predicted(expected, predicted)
        accurency, correct, incorrect = review_results(results)

        reviewed_results_info.append([len(predicted) - predicted.count(None), correct, incorrect, accurency])

    print_table(reviewed_results_info)
    print()
    print_result(results, expected)
