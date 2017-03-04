import tensorflow as tf
from utils import *
from blocks import *
import time

x = tf.placeholder(tf.float32, [None, 256 * 256 * 3])
keep_prob = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32, [None, 4])

y_conv = inference([
    ('conv', {'H': 3, 'F': 8, 'C': 3, 'stride': 4}),
    ('relu', {}),
    ('pool', {}),
    ('norm', {}),
    ('conv', {'H': 3, 'F': 16, 'C': 8, 'stride': 2}),
    ('relu', {}),
    ('norm', {}),
    ('pool', {}),
    ('fc', {'N': 8 * 8 * 16, 'M': 1024}),
    ('norm', {}),
    ('relu', {}),
    ('dropout', {'keep_prob': keep_prob}),
    ('fc', {'N': 1024, 'M': 200}),
    ('norm', {}),
    ('relu', {}),
    ('dropout', {'keep_prob': keep_prob}),
    ('fc', {'N': 200, 'M': 4})
], x, 256, 3)

cross_entropy, points, points_ = loss(y_conv, y, 'location')

train_step = train(cross_entropy, 1e-1)

# get the data
images = get_data("annotated/data.pkl")

# train
FLAGS = {'scratch': 'yes', 'log_dir': 'cropped_log'}
start_from = 0
steps = 1000

saver = tf.train.Saver()
sess = tf.Session()

# start from scratch or load a model
start_time = time.time()

with sess.as_default():

    sess.run(tf.global_variables_initializer())
    if FLAGS['scratch'] == 'no':
        saver.restore(sess, FLAGS['log_dir'] + "/model.ckpt-%d" % (start_from - 1))

    for i in range(start_from, steps+start_from):
        batch = images.train.next_batch(128)
        if i % 100 == 0:
            train_accuracy = sess.run(cross_entropy, feed_dict={
                x: batch[0], y: batch[1], keep_prob:1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob:0.5})

        if (i+1) % 1000 == 0:
            saver.save(sess, FLAGS['log_dir']+"/model.ckpt", global_step=i)

    out1 = open("out1.txt", "w+")
    out2 = open("out2.txt", "w+")
    ans = 0
    for i in range(len(images.test.X) // 16):
        predicted = points.eval(feed_dict={x: images.test.X[i* 16:(i+1)*16, :],
                                               y:images.test.y[i* 16:(i+1)*16, :],
                                               keep_prob: 1.0})
        actual = points_.eval(feed_dict={x: images.test.X[i* 16:(i+1)*16, :],
                                               y:images.test.y[i* 16:(i+1)*16, :],
                                               keep_prob: 1.0})

        acc = cross_entropy.eval(feed_dict={x: images.test.X[i * 16:(i + 1) * 16, :],
                                         y: images.test.y[i * 16:(i + 1) * 16, :],
                                         keep_prob: 1.0})
        print(acc)
        ans += acc

        out1.write(str(predicted))
        out2.write(str(actual))
    print(ans/(len(images.test.X) // 16))

    duration = time.time() - start_time
    print("Time: %.4f seconds" % duration)
