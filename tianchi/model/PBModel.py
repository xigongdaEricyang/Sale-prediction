import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# raw_df = pd.read_csv(r'C:\Users\YW59785\Desktop\tianchi\final_output\output_except_level_id.csv', delimiter=',').drop(['class_id'], axis=1)
raw_df = pd.read_csv(r'C:\Users\YW59785\Desktop\tianchi\final_output\final.csv', delimiter=',')
raw_df = raw_df.sample(frac=1).reset_index(drop=True)

normalizeColumns = ['compartment','TR','displacement','power',
                    'cylinder_number','engine_torque','car_length','car_height','car_width','total_quality','equipment_quality',
                    'rated_passenger','wheelbase','front_track','rear_track']

leftDf = raw_df.drop(normalizeColumns, axis =1 ).drop(['sale_quantity'], axis = 1)

normalizeDf = raw_df[normalizeColumns]
normalizeDf = (normalizeDf-normalizeDf.min())/(normalizeDf.max()-normalizeDf.min())
inputDf = pd.concat([leftDf, normalizeDf], axis = 1)
inputX = inputDf.values
resultArray = raw_df['sale_quantity'].values
inputY = resultArray.reshape((len(resultArray),1))

raw_set = raw_df.values

# scaler = MinMaxScaler()

# scaler.fit(raw_df.values[:, 1:20])

# raw_set = np.concatenate((raw_df.values[:, :1], np.concatenate((scaler.transform(raw_df.values[:, 1:20]), raw_df.values[:, 20:]), axis =1)), axis =1)

X, y= raw_set[:,1:], raw_set[:,0]

X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.33, random_state=42)
lr = 0.1
nepoch = 10000
batch_size = 2000

input_numbers = len(raw_df.columns.values)-1
first_hidden_layer_numbers = 40
second_hidden_layer_numbers = 20

# REGULARIZATION_RATE = 0.0001

graph = tf.Graph()
with graph.as_default():
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    x = tf.placeholder(tf.float32, [None, input_numbers])
    W_1 = tf.Variable(tf.truncated_normal([input_numbers, first_hidden_layer_numbers], stddev=0.1))
    b_1 = tf.Variable(tf.zeros([first_hidden_layer_numbers]))

    h_1 = tf.nn.relu(tf.add(tf.matmul(x, W_1), b_1))

    W_2 = tf.Variable(tf.truncated_normal([first_hidden_layer_numbers, second_hidden_layer_numbers], stddev=0.1))
    b_2 = tf.Variable(tf.zeros([second_hidden_layer_numbers]))

    h_2 = tf.nn.relu(tf.add(tf.matmul(h_1, W_2), b_2))

    W_3 = tf.Variable(tf.truncated_normal([second_hidden_layer_numbers, 1], stddev=0.1))
    b_3 = tf.Variable(tf.zeros([1]))

    y = tf.nn.relu(tf.add(tf.matmul(h_2, W_3), b_3))

    y_ = tf.placeholder(tf.float32, [None, 1])

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
    # cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y),
    #                 reduction_indices=[1]))
    loss = tf.reduce_mean(tf.abs(tf.subtract(y_, y)))
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    mae = tf.reduce_mean(tf.abs(y_ - y))
    # rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))
    # mse = tf.reduce_mean(tf.squared_difference(y_,y))

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
            print('mae: %.9f'%mae.eval(feed_dict=feed_dict))
        print('epoch mae: %.9f'%mae.eval(feed_dict={x: X_test, y_: y_test}))
        print('-----------------------------------------------------------')
        
