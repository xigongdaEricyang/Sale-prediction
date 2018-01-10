import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import normalize

raw_df = pd.read_csv(r'C:\Users\YW59785\Desktop\tianchi\final_output\output_except_level_id.csv', delimiter=',').drop(['class_id'], axis=1)

raw_set = raw_df.values

# raw_set = np.concatenate((raw_df.values[:, :1], np.concatenate((normalize(raw_df.values[:, 1:20]), raw_df.values[:, 20:]), axis =1)), axis =1)

X, y= raw_set[:,1:], raw_set[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
lr = 0.1
nepoch = 2000
batch_size = 20

input_numbers = len(raw_df.columns.values)-1
first_hidden_layer_numbers = 40
second_hidden_layer_numbers = 20

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, input_numbers])
    W_1 = tf.Variable(tf.random_normal([input_numbers, first_hidden_layer_numbers]))
    b_1 = tf.Variable(tf.zeros([first_hidden_layer_numbers]))

    h_1 = tf.nn.tanh(tf.add(tf.matmul(x, W_1), b_1))

    W_2 = tf.Variable(tf.random_normal([first_hidden_layer_numbers, second_hidden_layer_numbers]))
    b_2 = tf.Variable(tf.zeros([second_hidden_layer_numbers]))

    h_2 = tf.nn.tanh(tf.add(tf.matmul(h_1, W_2), b_2))

    W_3 = tf.Variable(tf.random_normal([second_hidden_layer_numbers, 1]))
    b_3 = tf.Variable(tf.zeros([1]))

    y = tf.nn.tanh(tf.add(tf.matmul(h_2, W_3), b_3))

    y_ = tf.placeholder(tf.float32, [None])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.transpose(y))))
    # cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y),
    #                 reduction_indices=[1]))

    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
    mse = tf.reduce_mean(tf.squared_difference(y_,y))

    # square_difference = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices=[1]))

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for epoch in range(nepoch):
        for start in range(0, len(X_train), batch_size):
            end = start + batch_size
            batch = X_train[start: end]
            y_batch = y_train[start: end]
            feed_dict = {x: batch, y_: y_batch}
            sess.run(optimizer, feed_dict=feed_dict)
        print('mse: %.9f'%mse.eval(feed_dict={x: X_test, y_: y_test}))
