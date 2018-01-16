import tensorflow as tf
import mnist_inference
import os
import pandas as pd
from sklearn.preprocessing import normalize
BATCH_SIZE = 100 
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH = "TianChi_Model/"
MODEL_NAME = "tianchi_model"

def train(trainX, trainY):
    dataSize = len(trainY)
    # 定义输入输出placeholder。
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    
    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
#     cross_entropy_mean = tf.reduce_mean(cross_entropy)
    beginLoss = tf.reduce_mean(tf.subtract(y, y_))
    loss = beginLoss + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        dataSize / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        head = 0
        for i in range(TRAINING_STEPS):
            tail = head+BATCH_SIZE
            if tail > dataSize:
                xs = trainX[head: BATCH_SIZE] + trainX[0: tail-BATCH_SIZE]
                ys = trainY[head: BATCH_SIZE] + trainY[0: tail-BATCH_SIZE]
                head = tail - BATCH_SIZE
            else:
                xs, ys = trainX[head: head+BATCH_SIZE-1], trainY[head: head+BATCH_SIZE-1]
                head = tail
            
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After", step," training step(s), loss on training batch is ", loss_value)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

df = pd.read_csv(r'C:\Users\YW59785\Desktop\tianchi\aa\final.csv')
# random shift the df
df = df.sample(frac=1).reset_index(drop=True)

normalizeColumns = ['compartment','TR','displacement','price_level','power',
                    'cylinder_number','engine_torque','car_length','car_height','total_quality','equipment_quality',
                    'rated_passenger','wheelbase','front_track','rear_track']
leftDf = df.drop(normalizeColumns, axis =1 ).drop(['sale_quantity'], axis = 1)

normalizeDf = df[normalizeColumns ]
inputDf = pd.concat([leftDf, normalizeDf])
inputX = (((inputDf-inputDf.min())/(inputDf.max()-inputDf.min())).values).tolist()
resultArray = df['sale_quantity'].values
inputY = (resultArray.reshape((len(resultArray),1))).tolist()

train(inputX, inputY)