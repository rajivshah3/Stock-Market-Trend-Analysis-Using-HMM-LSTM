from dataset_code.process_on_raw_data import form_raw_dataset
from public_tool.form_model_dataset import form_model_dataset
from public_tool.solve_on_outlier import solve_on_outlier
from train_model.GMM_HMM import GMM_HMM
from train_model.XGB_HMM import XGB_HMM
import pickle
from dataset_code import HMM_market_data, HMM_multi_factor
import os


def train_HMM_model(n_states):
    # train the market data or the multi factor GMM_HMM model and XGB_HMM model

    # 1 market data
    # 1.1 generate the market data dataset
    if not (os.path.exists('C:/Users/Administrator/Desktop/HMM_program/save/market_data_GMM_HMM_model.pkl') and os.path.exists('C:/Users/Administrator/Desktop/HMM_program/save/market_data_XGB_HMM_model.pkl')):
        feature_col = ['preClosePrice', 'openPrice', 'closePrice', 'turnoverVol', 'highestPrice', 'lowestPrice']
        dataset, label, lengths, col_nan_record = form_raw_dataset(feature_col, label_length=5)
        solved_dataset, allow_flag = HMM_market_data.solve_on_raw_data(dataset, lengths, feature_col)
        X_train, y_train, lengths_train = form_model_dataset(solved_dataset, label, allow_flag, lengths)
        X_train = solve_on_outlier(X_train, lengths_train)

        # 1.2 train and save the GMM_HMM model
        print('training market data GMM_HMM model...')
        temp = GMM_HMM(X_train, lengths_train, n_states, 'diag', 1000, True)
        pickle.dump(temp, open('C:/Users/Administrator/Desktop/HMM_program/save/market_data_GMM_HMM_model.pkl', 'wb'))

        # 1.3 train and save the XGB_HMM model
        print('training market data XGB_HMM model...')
        A, xgb_model, pi = XGB_HMM(X_train, lengths_train)
        pickle.dump([A, xgb_model, pi], open('C:/Users/Administrator/Desktop/HMM_program/save/market_data_XGB_HMM_model.pkl', 'wb'))

    # 2 multi factor
    print('training multi factor...')
    if not (os.path.exists('C:/Users/Administrator/Desktop/HMM_program/save/multi_factor_GMM_HMM_model.pkl') and os.path.exists('C:/Users/Administrator/Desktop/HMM_program/save/multi_factor_XGB_HMM_model.pkl')):
        score, feature_name = HMM_multi_factor.load_multi_factor_single_score()
        feature_col_multi_factor = HMM_multi_factor.type_filter(score, feature_name, 0.1)  # there are 7 kinds of multi factor
        GMM_model_list = []
        XGB_model_list = []
        for i in range(len(feature_col_multi_factor)):
            feature_col = feature_col_multi_factor[i]
            # 2.1 generate the multi factor dataset
            dataset, label, lengths, col_nan_record = form_raw_dataset(feature_col, label_length=5)
            print(sum(lengths))
            print(dataset.shape[0])
            solved_dataset, allow_flag = HMM_multi_factor.solve_on_raw_data(dataset, lengths, feature_col, feature_col)
            X_train, label_train, lengths_train = form_model_dataset(solved_dataset, label, allow_flag, lengths)
            pickle.dump([X_train, lengths_train], open('C:/Users/Administrator/Desktop/HMM_program/save/temp.pkl', 'wb'))
            X_train = solve_on_outlier(X_train, lengths_train)

            # 2.2 train and record the GMM_HMM model
            print('training multi factor GMM_HMM model %s...' % (i+1))
            pickle.dump([X_train, lengths_train], open('C:/Users/Administrator/Desktop/HMM_program/save/temp1.pkl', 'wb'))
            print(X_train.shape[0])
            print(sum(lengths_train))
            temp = GMM_HMM(X_train, lengths_train, n_states, 'diag', 1000, True)
            GMM_model_list.append(temp)

            # 2.3 train and record the XGB_HMM model
            print('training multi factor XGB_HMM model %s...' % (i+1))
            A, xgb_model, pi = XGB_HMM(X_train, lengths_train)
            XGB_model_list.append([A, xgb_model, pi])

        # 2.4 save the model
        pickle.dump(GMM_model_list, open('C:/Users/Administrator/Desktop/HMM_program/save/multi_factor_GMM_HMM_model.pkl', 'wb'))
        pickle.dump(XGB_model_list, open('C:/Users/Administrator/Desktop/HMM_program/save/multi_factor_XGB_HMM_model.pkl', 'wb'))
