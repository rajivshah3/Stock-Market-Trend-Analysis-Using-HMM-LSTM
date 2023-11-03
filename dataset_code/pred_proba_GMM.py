import numpy as np
from public_tool.form_index import form_index


def pred_proba_GMM(model, O, allow_flag, lengths):
    # Form pred_proba for the dataset. Note that the dataset here is the result of solve_on_raw_data, that is, the data with allow_flag.
    # output:
    #   pred_proba: array type

    pred_proba = np.zeros((O.shape[0], model.n_components))

    for i in range(len(lengths)):
        begin_index, end_index = form_index(lengths, i)

        now_O = O[begin_index:end_index, :]
        now_allow_flag = allow_flag[begin_index:end_index]
        now_pred_proba = np.zeros((now_O.shape[0], model.n_components))

        now_pred_proba[now_allow_flag == 1] = model.predict_proba(now_O[now_allow_flag == 1])

        pred_proba[begin_index:end_index] = now_pred_proba

    return pred_proba
