import pandas as pd


def read_dataset(path):
    data = pd.read_csv(path, sep=',', dtype=str, header=None)
    data.sample(frac=1)

    return data


def split_data(data, ratio_first_number, ratio_second_number):
    num_rows = data.shape[0]
    split_point = int(num_rows * ratio_first_number / (ratio_first_number + ratio_second_number))

    training_data = data[:split_point]
    testing_data = data[split_point:]

    return training_data, testing_data
