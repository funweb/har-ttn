import shutil

import tensorflow as tf
import numpy as np
import network_ttn_synthetic as network
import os.path
import scipy.io

from configuration import configure
from configuration.general import placeholder_inputs, load_data

from configuration import general


def train_ttn(dict_cus):
    datadir = os.path.join(configure.parameters_dict["database_dir"], 'ende',
                           dict_cus["dataset_name"],
                           str(dict_cus["distance_int"]),
                           'npy')
    cutdatadir = os.path.join(datadir, str(configure.parameters_dict["ksplit"]))

    for k in range(configure.parameters_dict["ksplit"]):
        #TODO: load data
        x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = load_data(
                dict_cus["dataset_name"],
                cutdatadir,
                dict_cus["seq_len"],
                k)

        train_data = x_train
        train_label = y_train  # 需要为 onehot 编码格式
        test_data = x_test
        test_label = y_test  # 需要为 onehot 编码格式

        num_classes = nb_classes

        configure.parameters_dict["num_classes"] = num_classes
        num_train = configure.parameters_dict["num_train"] = len(train_data)

        configure.parameters_dict = general.Merge(configure.parameters_dict, dict_cus)  # TODO: 不确定对不对

        batch_size = configure.parameters_dict["batch_size"]
        learning_rate_1 = configure.parameters_dict["learning_rate_1"]
        numBatches = int(num_train / batch_size)
        maxIters = configure.parameters_dict["maxIters"]
        num_classes = configure.parameters_dict["num_classes"]  # TODO: 类别数量, 根据需要设置
        seq_len = configure.parameters_dict["seq_len"]


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
            c_loss = 9999  # 初始化 loss
            train_loss = []

            for step in range(maxIters*numBatches+1):
                batchIdx = batchIdx % numBatches

                if batchIdx == 0:
                    randIdx = np.random.permutation(train_data.shape[0])

                trainIdx = randIdx[int(batchIdx * batch_size):int((batchIdx + 1) * batch_size)]

                if len(trainIdx) != batch_size:
                    print("trainIdx: {}".format(len(trainIdx)))
                    continue

                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={x_placeholder: train_data[trainIdx, :],
                                                    y_placeholder: train_label[trainIdx, :],
                                                    learning_rate_placeholder: learning_rate_1})
                train_loss.append(loss_value)

                if step % 1000 == 0:
                #     print('---------------------------------')
                #     print(step)
                #     print(loss_value)
                    print("k: {}\tstep: {}/{}, loss_value: {}".format(k, step, maxIters*numBatches, loss_value))

                batchIdx = batchIdx + 1
                if step in np.linspace(0, maxIters*numBatches, 6):
                    # weights_dir = os.path.join("weights")
                    weights_dir = general.getWeightsDir(configure.parameters_dict, k)
                    general.create_folder(weights_dir, remake=True)
                    weights_name = os.path.join(weights_dir, "%s_%s_gaussians_github_ttn_%s" % (str(k), str(step), str(loss_value)))
                    saver.save(sess, weights_name)
                    print(general.colorstr("model saved at: {}".format(weights_name)))

                    if loss_value < c_loss:
                        weights_name = os.path.join(weights_dir, "%s_best_model" % (str(k)))
                        saver.save(sess, weights_name)
                        print(general.colorstr("model saved at: {}".format(weights_name)))

            weights_name = os.path.join(weights_dir, "%s_last_model" % (str(k)))
            saver.save(sess, weights_name)
            print(general.colorstr("model saved at: {}".format(weights_name)))
            log_name = os.path.join(weights_dir, "%s_train_loss.txt" % (str(k)))
            np.savetxt(log_name, np.array(train_loss, dtype=np.float), delimiter=', ')
            print(general.colorstr("log saved at: {}".format(log_name)))

if __name__ == '__main__':
    dict_cus = {
        "batch_size": 32,
        "maxIters": 20,  # 100000
        "seq_len": 1024,
        "distance_int": 9999,
        "dataset_name": "cairo",
    }
    train_ttn(dict_cus)
