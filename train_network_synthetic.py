import tensorflow as tf
import numpy as np
import network_ttn_synthetic as network
import os.path
import scipy.io

from configuration import configure
from configuration.general import placeholder_inputs, load_data


def train_ttn():
    batch_size = configure.parameters_dict["batch_size"]
    learning_rate_1 = configure.parameters_dict["learning_rate_1"]
    num_train = configure.parameters_dict["num_train"]
    numBatches = num_train / batch_size
    maxIters = configure.parameters_dict["maxIters"]
    num_classes = configure.parameters_dict["num_classes"]  #TODO: 类别数量, 根据需要设置
    seq_len = configure.parameters_dict["seq_len"]


    datadir = os.path.join(configure.parameters_dict["database_dir"], 'ende',
                               configure.parameters_dict["dataset_name"],
                               str(configure.parameters_dict["distance_int"]),
                               'npy')
    cutdatadir = os.path.join(datadir, str(configure.parameters_dict["ksplit"]))

    for k in range(configure.parameters_dict["ksplit"]):
        #TODO: load data
        x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = load_data(
                configure["dataset_name"],
                cutdatadir,
                configure.parameters_dict["seq_len"],
                k)

        train_data = x_train
        train_label = y_train  # 需要为 onehot 编码格式
        num_classes = nb_classes

        with tf.Graph().as_default():
            x_placeholder, y_placeholder, learning_rate_placeholder = placeholder_inputs(batch_size, num_classes, configure.parameters_dict["seq_len"])
            output, sequence_unwarped, gamma, sequence1 = network.mapping(x_placeholder, batch_size, configure.parameters_dict["seq_len"])
            loss = network.loss(output, y_placeholder)

            var_list_ttn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ttn')
            var_list_classifier = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

            train_op_classifier = network.training(loss, learning_rate_placeholder, var_list_classifier)
            train_op_ttn = network.training(loss, learning_rate_placeholder / 10.0, var_list_ttn)
            train_op = tf.group(train_op_classifier, train_op_ttn)

            # train_op = network.training(loss, learning_rate_placeholder)

            init = tf.initialize_all_variables()
            saver = tf.train.Saver(max_to_keep=500)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            sess.run(init)

            batchIdx = 0
            for step in range(maxIters):
                batchIdx = batchIdx % numBatches

                if batchIdx == 0:
                    randIdx = np.random.permutation(train_data.shape[0])

                trainIdx = randIdx[int(batchIdx * batch_size):int((batchIdx + 1) * batch_size)]

                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={x_placeholder: train_data[trainIdx, :],
                                                    y_placeholder: train_label[trainIdx, :],
                                                    learning_rate_placeholder: learning_rate_1})

                if step % 1000 == 0:
                    print('---------------------------------')
                    print(step)
                    print(loss_value)

                batchIdx = batchIdx + 1

            saver.save(sess, './2_gaussians_github_ttn')
            print("success...")


if __name__ == '__main__':
    train_ttn(dict)
