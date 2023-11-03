import sys
import numpy as np
from public_tool.form_index import form_index


def combine(X1, X2, allow_flag1, allow_flag2, label, lengths):
    # Synthesize data from previous data that can be used later
    # Merge two X's
    # Samples with 0 in allow_flag are removed, and samples with -2 in label are removed.
    # input:
    #   X: The data to be merged can be in the format of 1 array or a list of multiple arrays.
    #   allow_flag: A flag that represents whether it is available. The format can be an array or a list of multiple arrays, corresponding to the X above.
    #   label: label
    #   lengths: lengths
    # output:
    #   result_X: new X matrix, array type
    #   result_label: new label
    #   result_lengths: new lengths

    if not (type(X1) == type(allow_flag1) or type(X2) == type(allow_flag2)):
        sys.exit('x 和 allow_flag的输入格式不一致')

    list_flag1 = type(X1) == list
    list_flag2 = type(X2) == list

    X = np.zeros((len(label), 0))
    allow_flag = np.zeros(len(label))
    count = 0

    if list_flag1 == 1:
        for i in range(len(X1)):
            X = np.column_stack(X, X1[i])
            allow_flag += allow_flag1[i]
            count += 1
    else:
        X = np.column_stack((X, X1))
        allow_flag += allow_flag1
        count += 1
    if list_flag2 == 1:
        for i in range(len(X2)):
            X = np.column_stack((X, X2[i]))
            allow_flag += allow_flag2[i]
            count += 1
    else:
        X = np.column_stack((X, X2))
        allow_flag += allow_flag2
        count += 1
    allow_flag[allow_flag < count] = 0
    allow_flag[allow_flag == count] = 1

    result_X = np.zeros((0, X.shape[1]))
    result_label = np.zeros(0)
    result_lengths = []

    for i in range(len(lengths)):
        begin_index, end_index = form_index(lengths, i)

        now_X = X[begin_index:end_index]
        now_allow_flag = allow_flag[begin_index:end_index]
        now_label = label[begin_index:end_index]

        temp = np.logical_and(now_allow_flag == 1, now_label != -2)

        result_X = np.row_stack((result_X, now_X[temp]))
        result_label = np.hstack((result_label, now_label[temp]))
        result_lengths.append(sum(temp))

    return result_X, result_label, result_lengths
