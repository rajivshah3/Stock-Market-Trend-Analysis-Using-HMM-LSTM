import sys
import numpy as np


def combine_allow_flag(allow_flag1, allow_flag2):
    # Combine the two allow flags, that is, they must be 1 only if they are 1 at the same time.

    if not len(allow_flag1) == len(allow_flag2):
        sys.exit('length of two allow_flag is not equal')
    result = np.zeros(len(allow_flag1))
    for i in range(len(allow_flag1)):
        if allow_flag1[i] == 1 and allow_flag2[i] == 1:
            result[i] = 1
        else:
            result[i] = 0
    return result
