import os, emoji
import numpy as np
import sklearn
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
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate, 如果是单变量
        # add a dimension to make it multivariate with one dimension  添加一个维度，使其具有一个维度的多变量
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # return x_train, y_train, x_train, y_train, y_true_train, nb_classes, y_true_train, enc
    return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc
    # ---


import tensorflow as tf

def placeholder_inputs(batch_size, num_classes, seq_len=2000):
    x_placeholder = tf.placeholder(tf.float32, shape=(batch_size, seq_len))
    y_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
    learning_rate_placeholder = tf.placeholder(tf.float32, shape=())
    return x_placeholder, y_placeholder, learning_rate_placeholder


