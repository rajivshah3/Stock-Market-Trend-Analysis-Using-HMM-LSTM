import numpy as np
import random


def bagging_balance_weight(X, y):
    # Solve the imbalanced data in the classification data set and baggage new data
    # input:
    #
    #   The X entered here can be two or three dimensions, as long as the first dimension is sample_num
    #   y can be a column vector or a one_hot encoded multi-column matrix
    # output:
    #   X_result, y_result
    #   The output format is the same as the input X, y

    drop_th = 0.01  # When the ratio of a certain class is less than this threshold, it is considered that there is no such class
    max_subsample_ratio = 1  # First record the largest max_n_subsample in the original data set, and then the number of each category in the bagging data set is max_n_subsample*this parameter

    if y.ndim == 1:
        y_label = y
    else:
        y_label = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            y_label[i] = np.where(y[i] == 1)[0][0]

    unique = np.unique(y_label)
    num_class = len(unique)
    unique_ratio = np.zeros(num_class)
    for i in range(num_class):
        unique_ratio[i] = sum(y_label == unique[i]) / len(y_label)

    unique_ratio[unique_ratio < drop_th] = 0

    n_bagging = int(max(unique_ratio) * len(y) * max_subsample_ratio)

    X_result = []
    y_result = []
    for i in range(num_class):
        if unique_ratio[i] == 0:
            continue
        else:
            sub_X = X[y_label == unique[i]]
            sub_y = y[y_label == unique[i]]
            for j in range(n_bagging):
                index = random.randint(0, sub_X.shape[0] - 1)
                X_result.append(sub_X[index])
                y_result.append(sub_y[index])
    X_result = np.array(X_result)
    y_result = np.array(y_result)
    # shuffle the order
    temp = [i for i in range(X_result.shape[0])]
    random.shuffle(temp)
    X_result = X_result[temp]
    y_result = y_result[temp]

    return X_result, y_result
