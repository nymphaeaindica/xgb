import pandas as pd
import numpy as np
from xml.etree import ElementTree as et

# column index of targetTrackID
target_id_col = 2


def get_vector(input_str, dtype):
    """Convert input string to list of float/integer variables."""
    l = input_str.split(' ')
    if dtype == 'float':
        output = list(map(float, l))
    elif dtype == 'int':
        output = list(map(int, l))
    return output


def get_str_vector(input_str):
    """Split input string by space delimiter."""
    return input_str.split(' ')


def parsconf(xml_path):
    """
    Reader for .xml configuration file.
    Returns dictionaries with parameters for XGBoost model.
    """
    root = et.parse(xml_path).getroot()

    settings = {}
    for child in root:
        for tmp in child:
            settings[tmp.tag] = tmp.attrib['value']

    # output dicts <-----
    # global params
    global_params = {}
    global_params['working_dir'] = settings.pop('working_dir')
    global_params['csv_file_name'] = settings.pop('csv_file_name')
    global_params['train_aux_file_name'] = settings.pop('train_aux_file_name')
    global_params['test_aux_file_name'] = settings.pop('test_aux_file_name')
    global_params['output_model_file_name'] = settings.pop('output_model_file_name')

    # external params
    external_params = {}
    # external_params['kfold'] = get_vector(settings.pop('kfold'))
    external_params['kfold'] = int(settings.pop('kfold'))
    # external_params['num_boost_round'] = get_vector(settings.pop('num_boost_round'))
    external_params['num_boost_round'] = int(settings.pop('num_boost_round'))
    # external_params['early_stopping_rounds'] = get_vector(settings.pop('early_stopping_rounds'))
    external_params['early_stopping_rounds'] = int(settings.pop('early_stopping_rounds'))
    external_params['best_model_count'] = int(settings.pop('best_model_count'))
    external_params['threshold'] = get_vector(settings.pop('threshold'), 'float')

    eta = get_vector(settings.pop('eta'), 'float')
    gamma = get_vector(settings.pop('gamma'), 'float')
    max_depth = get_vector(settings.pop('max_depth'), 'int')
    min_child_weight = get_vector(settings.pop('min_child_weight'), 'float')
    max_delta_step = get_vector(settings.pop('max_delta_step'), 'float')
    subsample = get_vector(settings.pop('subsample'), 'float')
    colsample_bytree = get_vector(settings.pop('colsample_bytree'), 'float')
    lambda_ = get_vector(settings.pop('lambda'), 'float')
    alpha = get_vector(settings.pop('alpha'), 'float')
    tree_method = get_str_vector(settings.pop('tree_method'))
    # nthread = get_vector(settings.pop('nthread'))
    nthread = int(settings.pop('nthread'))
    scale_pos_weight = get_vector(settings.pop('scale_pos_weight'), 'float')
    objective = get_str_vector(settings.pop('objective'))
    eval_metric = get_str_vector(settings.pop('eval_metric'))
    # seed = get_vector(settings.pop('seed'))
    seed = int(settings.pop('seed'))

    param_list = []
    for par1 in eta:
        for par2 in gamma:
            for par3 in max_depth:
                for par4 in min_child_weight:
                    for par5 in max_delta_step:
                        for par6 in subsample:
                            for par7 in colsample_bytree:
                                for par8 in lambda_:
                                    for par9 in alpha:
                                        for par10 in tree_method:
                                            # for par11 in nthread:
                                                for par12 in scale_pos_weight:
                                                    for par13 in objective:
                                                        for par14 in eval_metric:
                                                            # for par15 in seed:
                                                                booster_params = {}
                                                                booster_params['eta'] = par1
                                                                booster_params['gamma'] = par2
                                                                booster_params['max_depth'] = par3
                                                                booster_params['min_child_weight'] = par4
                                                                booster_params['max_delta_step'] = par5
                                                                booster_params['subsample'] = par6
                                                                booster_params['colsample_bytree'] = par7
                                                                booster_params['lambda'] = par8
                                                                booster_params['alpha'] = par9
                                                                booster_params['tree_method'] = par10
                                                                # booster_params['nthread'] = par11
                                                                booster_params['nthread'] = nthread
                                                                booster_params['scale_pos_weight'] = par12
                                                                booster_params['objective'] = par13
                                                                booster_params['eval_metric'] = par14
                                                                # booster_params['seed'] = par15
                                                                booster_params['seed'] = seed
                                                                param_list.append(booster_params)
    return global_params, external_params, param_list


def read_aux(aux, features_index):
    """
    Reader for .aux file.
    Takes necessary features, labels and targets from .dat files
    into numpy arrays.
    """
    path_list = aux[1]
    label_list = aux[0].to_numpy()
    # init output values
    features = np.empty([0, len(features_index)])
    labels = np.empty([0, 1])
    target_id = np.empty([0, 1])
    max_val = 0

    for ind, path in enumerate(path_list):
        data = pd.read_csv(path, header=None, sep=' ').to_numpy()
        data = np.unique(data[:, :-2], axis=0)
        tmp_target_id = data[:, [target_id_col]]
        target_id = np.concatenate((target_id, tmp_target_id + max_val), axis=0)
        max_val = max(target_id) + 1
        tmp_features = data[:, features_index]
        features = np.concatenate((features, tmp_features), axis=0)
        if label_list[ind] == '1':
            tmp_labels = np.ones([tmp_features.shape[0], 1])
        else:
            tmp_labels = np.zeros([tmp_features.shape[0], 1])
        labels = np.concatenate((labels, tmp_labels), axis=0)

    return features, labels, target_id


def parsdata(global_params):
    """
    Split data into train and test datasets.
    """
    # datatype: string 'train' or 'test'
    configcsv = pd.read_csv(global_params['working_dir'] + global_params['csv_file_name'], header=0, usecols=[1, 17]).query('column18 != 0')
    # configcsv = configcsv.query('column18 != 0')
    features_index = configcsv.index
    aux_train = pd.read_csv(global_params['working_dir'] + global_params['train_aux_file_name'], header=None, sep=',', dtype=str)
    aux_test = pd.read_csv(global_params['working_dir'] + global_params['test_aux_file_name'], header=None, sep=',', dtype=str)
    features_train, labels_train, targetID_train = read_aux(aux_train, features_index)
    features_test, labels_test, targetID_test = read_aux(aux_test, features_index)
    train = {'features': features_train, 'labels': labels_train, 'target_id': targetID_train}
    test = {'features': features_test, 'labels': labels_test, 'target_id': targetID_test}
    return train, test
