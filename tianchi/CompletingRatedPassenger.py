import pandas as pd
import numpy  as np
import math
import tensorflow as tf


raw_data_df = pd.read_csv(r'./tmp_data_files/rp_df.csv', sep=',')

rp_df = raw_data_df[['car_length', 'car_width', 'car_height', 'rated_passenger']]

def getDataSet(isAvaliable_set = True):
    tmp_set = rp_df[rp_df.rated_passenger.str.contains('\d+$') == isAvaliable_set]
    x_results = tmp_set.drop('rated_passenger', axis=1).values
    y_results = tmp_set['rated_passenger'].values
    return x_results, y_results

result_x_avi, result_y_avi = getDataSet() 

length = len(result_x_avi)
split = math.ceil(0.7*length)


train_set_x = result_x_avi[0: split]
train_set_y = result_y_avi[0: split]

test_set_x = result_x_avi[split:]
test_set_y = result_y_avi[split:]

result_x_des, result_y_des = getDataSet(False)

lr = 0.01
nepochs = 100
batch_size = 20
display_step = 500
# current_W = np.zeros([3], np.float)
# current_bias = np.zeros(np.float)

graph = tf.Graph()

with graph.as_default():
    x = tf.placeholder(tf.float64, [None, 3])
    W = tf.Variable(tf.zeros([3,1])) 
    bias = tf.Variable(tf.zeros([1]))
    y = tf.placeholder(tf.float64, [None, 1])
    y_prediction = tf.add(tf.matmul(tf.cast(x, tf.float64), tf.cast(W, tf.float64)),tf.cast(bias, tf.float64))
    #Loss
    cost = tf.reduce_sum(tf.pow(y_prediction - y, 2))/(2 * nepochs)
    # cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_prediction), reduction_indices=[1]))

    #optimizer
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for epoch in range(nepochs):
        for start in range(0, split-batch_size, batch_size):
            end = start + batch_size
            batch = train_set_x[start : end]
            y_dest = train_set_y[start : end].reshape(batch_size, 1)
            feed_dict = {x: batch, y: y_dest}
            sess.run(optimizer, feed_dict=feed_dict)
        # if epoch % display_step == 0:
        print('Epoch:{}, cost={}'.format((epoch +1), sess.run(cost, feed_dict={x: test_set_x, y: test_set_y})))
        