import tensorflow as tf
import numpy as np
import network_ttn_synthetic as network
import os.path
import scipy.io

from configuration import configure
from configuration.general import placeholder_inputs

batch_size = 1
seq_len = configure.parameters_dict["seq_len"]
num_classes = configure.parameters_dict["num_classes"]  # TODO: 类别数目

test_data = scipy.io.loadmat('synthetic_data_test_2_gaussians.mat')['test_data']
test_label = scipy.io.loadmat('synthetic_data_test_2_gaussians.mat')['test_label']

test_label = np.pad(test_label,((0,0), (1,0)),'constant',constant_values = (0,0))  #constant_values表示填充值，且(before，after)的填充值等于（0,0）

# TODO: load data



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
    generated_gamma = np.zeros((test_data.shape[0], 100))
    ttn_output = np.zeros((test_data.shape[0], 100))
    sequence_normalized = np.zeros((test_data.shape[0], 100))

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
