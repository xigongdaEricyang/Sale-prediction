import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# raw_df = pd.read_csv(r'C:\Users\YW59785\Desktop\tianchi\final_output\output_except_level_id.csv', delimiter=',').drop(['class_id'], axis=1)
# raw_set = raw_df.values
# X, y= raw_set[:,1:], raw_set[:,0]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

INPUT_NODE = 245    # 输入节点
OUTPUT_NODE = 1     # 输出节点
LAYER1_NODE = 500    # 隐藏层数       
                              
BATCH_SIZE = 200     # 每次batch打包的样本个数        

# 模型相关的参数
LEARNING_RATE_BASE = 0.8      
LEARNING_RATE_DECAY = 0.99    
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 2000        
MOVING_AVERAGE_DECAY = 0.99  

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)  


def train(trainX, trainY, X_test, y_test):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)
    
    # 定义训练轮数及相关的滑动平均类 
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    
    # 计算交叉熵及其平均值
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    # cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = tf.reduce_mean(tf.abs(y_ - y)) + regularaztion
    
    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        13000 / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    # correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    mae = tf.reduce_mean(tf.abs(y_ - y))
    
    # 初始化回话并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: X_test, y_: y_test} 
        
        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            for start in range(0, len(trainX), BATCH_SIZE):
                end = start + BATCH_SIZE
                if end < len(trainX):
                    x_batch = trainX[start: end]
                    y_batch = trainY[start: end]
                else:
                    x_batch = trainX[start: ]
                    y_batch = trainY[start: ]
                feed_dict = {x: x_batch, y_: y_batch}
                sess.run(train_op, feed_dict=feed_dict)
                # xs,ys=mnist.train.next_batch(BATCH_SIZE)
                # sess.run(train_op,feed_dict={x:xs,y_:ys})
            if i % 1000 == 0:
                validate_acc = sess.run(mae, feed_dict=test_feed)
                print("After %d training step(s), validation mae using average model is %g " % (i, validate_acc))
            


        test_acc=sess.run(mae,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(TRAINING_STEPS, test_acc)))


def main(argv=None):
    df = pd.read_csv(r'C:\Users\YW59785\Desktop\tianchi\aa\final.csv')
    # random shift the df
    df = df.sample(frac=1).reset_index(drop=True)

    normalizeColumns = ['compartment','TR','displacement','price_level','power','level_id',
                    'cylinder_number','engine_torque','car_length','car_height','car_width','total_quality','equipment_quality',
                    'rated_passenger','wheelbase','front_track','rear_track']
    leftDf = df.drop(normalizeColumns, axis =1 ).drop(['sale_quantity'], axis = 1)

    normalizeDf = df[normalizeColumns]
    normalizeDf = (normalizeDf-normalizeDf.min())/(normalizeDf.max()-normalizeDf.min())
    inputDf = pd.concat([leftDf, normalizeDf], axis = 1)
    inputX = inputDf.values
    resultArray = df['sale_quantity'].values
    inputY = resultArray.reshape((len(resultArray),1))
    # mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.33, random_state=42)
    train(X_train, y_train, X_test, y_test)

if __name__=='__main__':
    main()