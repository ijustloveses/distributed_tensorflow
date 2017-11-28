# encoding: utf-8

from __future__ import print_function
import tensorflow as tf
import sys
import time
from settings import ps_svrs, worker_svrs

cluster = tf.train.ClusterSpec({"ps": ps_svrs, "worker": worker_svrs})

tf.app.flags.DEFINE_string("job_name", "", "ps or worker")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index, config=config)

batch_size = 100
lr = 0.001
epochs = 100
logs_path = './logs'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

if FLAGS.job_name == 'ps':
    print("ps setting up ...")
    server.join()
elif FLAGS.job_name == 'worker':
    print("worker setting up ...")
    # BETWEEN-GRAP replication
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, FLAGS.task_index),
        cluster=cluster)):

        global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)

        # input images
        with tf.name_scope('input'):
            # None -> batch size can be any size, 784 -> flattened mnist image
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            # target 10 output classes
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            W1 = tf.Variable(tf.random_normal([784, 100]))
            W2 = tf.Variable(tf.random_normal([100, 10]))
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([100]))
            b2 = tf.Variable(tf.zeros([10]))

        with tf.name_scope("softmax"):
            z2 = tf.add(tf.matmul(x,W1),b1)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(a2,W2),b2)
            y  = tf.nn.softmax(z3)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        with tf.name_scope('train'):
            # Asynchronous training
            grad_op = tf.train.GradientDescentOptimizer(lr)
            train_op = grad_op.minimize(cross_entropy, global_step=global_step)

        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print("Ready to go")

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), global_step=global_step, init_op=init_op)

    begin_time = time.time()
    freq = 100

    with sv.prepare_or_wait_for_session(server.target) as sess:
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        start_time = time.time()
        for epoch in range(epochs):
            batch_count = int(mnist.train.num_examples / batch_size)
            count = 0
            
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, cost, summary, step = sess.run(
                    [train_op, cross_entropy, summary_op, global_step],
                    feed_dict={x: batch_x, y_: batch_y})
                writer.add_summary(summary, step)

                count += 1
                if count % freq == 0 or i + 1 == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (step+1), 
                        " Epoch: %2d," % (epoch+1), 
                        " Batch: %3d of %3d," % (i+1, batch_count), 
                        " Cost: %.4f," % cost, 
                        " AvgTime: %3.2fms" % float(elapsed_time*1000/freq))
                    count = 0

            print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            print("Total Time: %3.2fs" % float(time.time() - begin_time))
            begin_time = time.time()
            print("Final Cost: %.4f" % cost)

    sv.stop()
    print("Done")
