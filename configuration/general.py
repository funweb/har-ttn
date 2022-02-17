import os, emoji
import shutil
import time

import numpy as np
from sklearn import preprocessing
import tensorflow as tf

def load_data(data_name, cutdatadir, data_lenght=2000, k=2):
    """
    功能: 载入数据集, 可以按照约定的格式返回, 包括: 训练数据集, 测试数据集, 类别, 编码等;  # TODO: 类别编码问题估计还需要进一步调整
    可调整的包括: 控制数据的长度
    单变量: 是否将单变量转换为适用于 卷积 的格式
    """

    assert 2001 > data_lenght > 0, "Please check data_lenght: {} ".format(data_lenght)  # 长度要适合

    data_type = 'train'
    data_x_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-x-' + str(k) + '.npy')

    print(emoji.emojize(":check_mark_button: data_x_path: {}".format(data_x_path)))

    x_train = np.load(data_x_path, allow_pickle=True)

    data_y_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-y-' + str(k) + '.npy')
    y_train = np.load(data_y_path, allow_pickle=True)

    data_labels_path = os.path.join(cutdatadir, data_name + '-labels.npy')
    dictActivities_x = np.load(data_labels_path, allow_pickle=True).item()

    data_type = 'test'
    data_x_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-x-' + str(k) + '.npy')
    x_test = np.load(data_x_path, allow_pickle=True)

    data_y_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-y-' + str(k) + '.npy')
    y_test = np.load(data_y_path, allow_pickle=True)

    data_labels_path = os.path.join(cutdatadir, data_name + '-labels.npy')
    dictActivities_y = np.load(data_labels_path, allow_pickle=True).item()

    # x_range = len(np.unique(np.concatenate((x_train, x_test), axis=0)))  # 一共有多少种状态
    # x_train = x_train / (x_range + 1)  # 归一化处理  # TODO: 感觉没必要
    x_train = x_train[:, -data_lenght:]  # 控制数据长度
    # x_test = x_test / (x_range + 1)
    x_test = x_test[:, -data_lenght:]

    # ---
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))  # 类别数据, 也就是 len(dictActivities_y)

    # save orignal y because later we will use binary  保存原始y，因为稍后我们将使用二进制
    y_true = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)
    # transform the labels from integers to one hot vectors  将标签从整数转换为 one hot vectors
    enc = preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # if len(x_train.shape) == 2:  # if univariate, 如果是单变量
    #     # add a dimension to make it multivariate with one dimension  添加一个维度，使其具有一个维度的多变量
    #     x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    #     x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # return x_train, y_train, x_train, y_train, y_true_train, nb_classes, y_true_train, enc
    return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc
    # ---


import tensorflow as tf

def placeholder_inputs(batch_size, num_classes, seq_len=2000):
    x_placeholder = tf.placeholder(tf.float32, shape=(batch_size, seq_len))
    y_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
    learning_rate_placeholder = tf.placeholder(tf.float32, shape=())
    return x_placeholder, y_placeholder, learning_rate_placeholder




def create_folder(path='./new', remake=False):
    # Create folder
    if not os.path.exists(path):
        print('Create subdir directory: %s...' % (path))
        time.sleep(3)
        os.makedirs(path)
    elif remake:
        shutil.rmtree(path)  # delete output folder
        os.makedirs(path)


# Merge the two dictionaries, that is, the parameters required by the algorithm
def Merge(dict_config, dict_config_cus):
    dict_config.update(dict_config_cus)
    for k in dict_config:  # 保证值不为空, 也就是保证参数的有效性
        assert dict_config[k] != "", "Please set value for: {}".format(k)
    return dict_config


def reTrain(p):
    if os.path.exists(p):
        shutil.rmtree(p)


import urllib, json, os, ipykernel, ntpath
from notebook import notebookapp as app


def lab_or_notebook():
    length = len(list(app.list_running_servers()))
    if length:
        return "notebook"
    else:
        return "lab"


def ipy_nb_name(token_lists):
    """ Returns the short name of the notebook w/o .ipynb
        or get a FileNotFoundError exception if it cannot be determined
        NOTE: works only when the security is token-based or there is also no password
    """

    if lab_or_notebook() == "lab":
        from jupyter_server import serverapp as app
    else:
        from notebook import notebookapp as app
    #         from jupyter_server import serverapp as app

    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    #     from notebook import notebookapp as app
    for srv in app.list_running_servers():
        for token in token_lists:
            srv['token'] = token

            try:
                # print(token)
                if srv['token'] == '' and not srv['password']:  # No token and no password, ahem...
                    req = urllib.request.urlopen(srv['url'] + 'api/sessions')
                    print('no token or password')
                else:
                    req = urllib.request.urlopen(srv['url'] + 'api/sessions?token=' + srv['token'])
            except:
                pass
                # print("Token is error")

        sessions = json.load(req)

        for sess in sessions:
            if sess['kernel']['id'] == kernel_id:
                nb_path = sess['notebook']['path']
                return ntpath.basename(nb_path).replace('.ipynb', '')  # handles any OS

    raise FileNotFoundError("Can't identify the notebook name, Please check [token]")


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def getIdentify(dict_cus):
    identify = "{}_{}_{}".format(str(dict_cus["maxIters"]), str(dict_cus["batch_size"]), str(dict_cus["seq_len"]))
    return identify


def getWeightsDir(dict_cus, k):
    identify = getIdentify(dict_cus)
    weights_dir = os.path.join(dict_cus["database_dir"],
                 "results",
                 dict_cus["method_name"],
                 dict_cus["dataset_name"],
                 str(dict_cus["distance_int"]),
                 identify,
                 str(k))
    return weights_dir
