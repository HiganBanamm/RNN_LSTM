# All Includes

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics

import os

# Useful Constants

# 采集到的各种信号
# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

# os.chdir(DATA_PATH)
# os.chdir("..")

# print("\n" + "Dateset is now located at:" + DATASET_PATH)

# Preparing dataset

DATA_PATH = "data/"
DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
TRAIN = "train/"
TEST = "test/"


# 开始写：得到X_train, X_test,y_train,y_test数据集------------------------------------------------------------------------
# Load "X"(the neural network's training and testing inputs) 处理数据以及取数据
def load_X(X_signals_paths):
    X_signals = []
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')

        # 此处为双循环，先从file中取row，然后对row操作
        # 后再在row上取serie，最后转换
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        # for row in file:
        #     new_row = row.replace('  ', ' ').strip().split(' ') # 先把每行数据处理一下
        #     for serie in new_row:
        #         serie = serie.strip().split(' ')
        #         ss = np.array(serie, dtype=np.float32)  # 转换成一个数组
        #         X_signals.append([ss])

        file.close()
    # 矩阵的转置

    return np.transpose(np.array(X_signals), (1, 2, 0))  # 第一个参数是数组，第二个参数是把传进的这个数组转置成（1，2,0）形状


# X_train_signals_paths 就是每次经过for循环得到的数据集的路径
X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]

# 得到训练集数据所在的路径
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

# 调用load_X()函数，得到已经处理好的X_train
X_train = load_X(X_train_signals_paths)
print("X_train shape:", X_train.shape)

# 调用load_X()函数，得到已经处理好的X_test
X_test = load_X(X_test_signals_paths)
print("X_test shape: ", X_test.shape)


# Load "y"(the neural network's training and testing outputs)
def load_y(y_path):
    file = open(y_path, 'r')

    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    # Read dataset from disk,dealing with text file's syntax

    # for row in file:
    #     new_row = row.replace('  ', ' ').strip().split(' ')  # 先把文件中的每一行数据处理一下，删除空格和分隔符
    #     for elem in new_row:
    #         y_ = np.array(elem, dtype=np.int32)  # 然后处理后的数据转换成一个数组

    return y_ - 1
    file.close()

    # 为每个输出类减1，以实现友好的基于0的索引


y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
print("y_train", y_train)
print(np.array(y_train).shape)
y_test = load_y(y_test_path)
print("y_test", y_test)
print(np.array(y_test).shape)
# 以上就完成了得到X_train, X_test,y_train,y_test数据集---------------------------------------------------------------------


# Additionnal Parameters————————————————————————————————————————————————————————————————————————————————————————————————

# input data
training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep

# LSTM Neural Network's internal structure
n_hidden = 32
n_classes = 6

# training
learning_rate = 0.0025
lamba_loss_amount = 0.0015
training_iters = training_data_count * 300  # 每个数据集循环300次
batch_size = 1500
display_iter = 30000  # To show test set accuracy during training


# print(X_test.shape, '\n')
# print(y_test.shape, '\n')
# print(np.mean(X_test), '\n')
# print(np.std(X_test), '\n')


# Utility functions for training————————————————————————————————————————————————————————————————————————————————————————
def LSTM_RNN(_X, _weights, _biases):
    # 1.对输入的数据进行变形，转化成LSTM的输入数据的格式

    # input shape: (batch_size, n_steps, n_input) (7352, 128, 9)
    # (128, 7352, 9)
    _X = tf.transpose(_X, [1, 0, 2])

    # input shape: (batch_size*n_steps, n_input)
    # (941056, 9)
    _X = tf.reshape(_X, [-1, n_input])

    # (941056, 32)
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])

    # (7352, 32)
    _X = tf.split(value=_X, num_or_size_splits=n_steps, axis=0)  # value: tensor; num_or_size_splits:原数据集将有n_steps次划分

    # 2.Define two stacked LSTM cells (two recurrent layers deep) with tensorflow

    #  tf.nn.rnn_cell.BasicLSTMCell
    # args1: num_units：LSTM单元中的单元数；args2:forget_bias: float, The bias added to forget gates
    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0, state_is_tuple=True)
    # tf.contrib.rnn.MultiRNNCell()
    # 参数1：cells：将按此顺序组成的RNNCell列表
    # 参数2：state_is_tuple：如果为True，则接受和返回的状态是n元组，其中n = len(cells)
    lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1, lstm_cell_2], state_is_tuple=True)

    # Get LSTM cell output
    # contrib.rnn.static_rnn()
    # args1:cell：RNNCell的一个实例
    # args2:inputs：输入的长度T列表，每个Tensor形状[batch_size, input_size]或这些元素的嵌套元组

    # outputs --> shape=(7352, 32)
    # states --> shape=(7352, 32)
    outputs, states = tf.contrib.rnn.static_rnn(cell=lstm_cells, inputs=_X, dtype=tf.float32)

    # shape=(7352, 32)
    lstm_last_output = outputs[-1]

    # shape=(7352, 6)
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # function to fetch a "batch_size" amount of data from "(X|y)_train "data
    shape = list(_train.shape)  # [7352, 128, 9]

    shape[0] = batch_size # 7352

    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step - 1) * batch_size + i) % len(_train)  # len(_train)=7352
        batch_s[i] = _train[index]  # _train-->[[[1,2,3,4] [5,6,7,8] [9,10,11,12]],[[13,14,15,16] [17,18,19,20] [21,22,23,24]],[[25,26,27,28] [29,30,31,32] [33,34,35,36]]]
                                    # _train[0] 指的是[[1,2,3,4] [5,6,7,8] [9,10,11,12]]
                                    # _train[0][0] 指的是[1,2,3,4]
    return batch_s


def one__hot(y_, n_classes=n_classes):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0,     0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))  # 7352
    # np.eye(n_classes) 单位矩阵
    # --> shape=(7352, 6)
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # np.eye(6)[[4,4,4,5,5,6,4,3,7,1,1]] 分别取label[x]行的值


# build the neural network——————————————————————————————————————————————————————————————————————————————————————————————
# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # shape=(9,32)
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))  # shape=(32,6)
}

biases = {
    'hidden': tf.Variable(tf.random_normal(shape=[n_hidden])),  # shape=(32)
    'out': tf.Variable(tf.random_normal(shape=[n_classes]))  # shape=(6)
}

pred = LSTM_RNN(x, weights, biases)  # 返回的矩阵：shape=(7352, 6)

# loss, optimizer and evaluation
l2 = lamba_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    # # L2 loss prevents this overkill neural network to overfit the data
)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(input=pred, axis=1), tf.argmax(input=y, axis=1))  # argmax()返回最大值的索引，equal()判断是否相等

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast()将bool类型转化为0或1.true转化后是1，false转化后是0

# now train the neural network——————————————————————————————————————————————————————————————————————————————————————————

# To keep track of training's performance
test_losses = []
test_accuracied = []
train_losses = []
train_accuracied = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# 每次循环中使用“batch_size”的数据执行训练步骤
step = 1
while step * batch_size <= training_iters:  # training_iters = 7352 * 300
    batch_xs = extract_batch_size(X_train, step, batch_size)
    batch_ys = one__hot(extract_batch_size(y_train, step, batch_size))

    # fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy], feed_dict={
            x: batch_xs, y: batch_ys
        }
    )

    train_losses.append(loss)
    train_accuracied.append(acc)

    # 仅在某些步骤评估网络，以加快培训速度
    if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters): # training_iters = 7352 * 300
        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step * batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))

        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy],
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1
print("Optimization Finished!")

# Accuracy for test data

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))
