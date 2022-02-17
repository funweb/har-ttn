import tensorflow as tf
import numpy as np
import network_ttn_synthetic as network
import os.path
import scipy.io

from configuration import configure, general
from configuration.general import placeholder_inputs, load_data


def test_ttn(dict_cus):
    datadir = os.path.join(configure.parameters_dict["database_dir"], 'ende',
                           dict_cus["dataset_name"],
                           str(dict_cus["distance_int"]),
                           'npy')
    cutdatadir = os.path.join(datadir, str(configure.parameters_dict["ksplit"]))

    for k in range(configure.parameters_dict["ksplit"]):

        # TODO: load data
        x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = load_data(
            dict_cus["dataset_name"],
            cutdatadir,
            dict_cus["seq_len"],
            k)

        train_data = x_train
        train_label = y_train  # 需要为 onehot 编码格式
        num_classes = nb_classes

        configure.parameters_dict["num_classes"] = num_classes
        num_train = configure.parameters_dict["num_train"] = len(train_data)

        general.Merge(configure.parameters_dict, dict_cus)

        batch_size = configure.parameters_dict["batch_size"]
        learning_rate_1 = configure.parameters_dict["learning_rate_1"]
        numBatches = int(num_train / batch_size)
        maxIters = configure.parameters_dict["maxIters"]
        seq_len = configure.parameters_dict["seq_len"]

        batch_size = 1
        seq_len = dict_cus["seq_len"]


        num_classes = nb_classes
        test_data = x_train
        test_label = y_train
        test_data = x_test
        test_label = y_test

        # test_data = scipy.io.loadmat('synthetic_data_test_2_gaussians.mat')['test_data']
        # test_label = scipy.io.loadmat('synthetic_data_test_2_gaussians.mat')['test_label']

        with tf.Graph().as_default():
            checkpointPath = './2_gaussians_github_ttn'

            x_placeholder, y_placeholder, _ = placeholder_inputs(batch_size,  configure.parameters_dict["num_classes"], configure.parameters_dict["seq_len"])  # 类别数应该改为载入数据的格式
            output, sequence_unwarped, gamma, sequence1 = network.mapping(x_placeholder, batch_size, configure.parameters_dict["seq_len"])
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_placeholder, 1))

            saver = tf.train.Saver()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            saver.restore(sess, checkpointPath)

            correct = 0
            generated_gamma = np.zeros((test_data.shape[0], seq_len))
            ttn_output = np.zeros((test_data.shape[0], seq_len))
            sequence_normalized = np.zeros((test_data.shape[0], seq_len))

            for testExample in range(test_data.shape[0]):
                output_val, ttn_output_val, gamma_val, sequence1_val, correct_val = sess.run(
                    [output, sequence_unwarped, gamma, sequence1, correct_prediction],
                    feed_dict={x_placeholder: np.reshape(test_data[testExample, :], [1, seq_len]),
                               y_placeholder: np.reshape(test_label[testExample, :], [1, num_classes])})
                correct = correct + int(correct_val[0] == True)

                generated_gamma[testExample, :] = gamma_val
                ttn_output[testExample, :] = ttn_output_val
                sequence_normalized[testExample, :] = np.squeeze(sequence1_val)

        accuracy = correct / float(test_data.shape[0])

        scipy.io.savemat('./ttn_output_github.mat', mdict={'ttn_output': ttn_output})
        scipy.io.savemat('./generated_gamma_github.mat', mdict={'generated_gamma': generated_gamma})
        scipy.io.savemat('./sequence_normalized_github.mat', mdict={'sequence_normalized': sequence_normalized})

        print('Accuracy:')
        print(accuracy)

        break


if __name__ == '__main__':
    dict_cus = {
        "batch_size": 32,
        "maxIters": 1000,
        "seq_len": 512,
        "distance_int": 9999,
        "dataset_name": "cairo",
    }

    test_ttn(dict_cus)
