from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import os
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
FLAGS = None

def generate_embeddings():
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True,
                                      fake_data=FLAGS.fake_data)
    sess = tf.InteractiveSession()

    # Input set for Embedded TensorBoard visualization
    # Performed with cpu to conserve memory and processing power
    with tf.device("/cpu:0"):
        embedding = tf.Variable(tf.stack(mnist.test.images[:FLAGS.max_steps], axis=0), trainable=False, name='embedding')

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(FLAGS.log_dir + '/projector', sess.graph)

    # Add embedding tensorboard visualization. Need tensorflow version
    # >= 0.12.0RC0
    config = projector.ProjectorConfig()
    embed= config.embeddings.add()
    embed.tensor_name = 'embedding:0'
    embed.metadata_path = os.path.join(FLAGS.log_dir + '/projector/metadata.tsv')
    embed.sprite.image_path = os.path.join(FLAGS.data_dir + '/mnist_10k_sprite.png')

    # Specify the width and height of a single thumbnail.
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(writer, config)

    saver.save(sess, os.path.join(
        FLAGS.log_dir, 'projector/a_model.ckpt'), global_step=FLAGS.max_steps)

def generate_metadata_file():
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True,
                                      fake_data=FLAGS.fake_data)
    def save_metadata(file):
        with open(file, 'w') as f:
            for i in range(FLAGS.max_steps):
                c = np.nonzero(mnist.test.labels[::1])[1:][0][i]
                f.write('{}\n'.format(c))

    save_metadata(FLAGS.log_dir + '/projector/metadata.tsv')

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir + '/projector'):
        tf.gfile.DeleteRecursively(FLAGS.log_dir + '/projector')
        tf.gfile.MkDir(FLAGS.log_dir + '/projector')
    tf.gfile.MakeDirs(FLAGS.log_dir  + '/projector') # fix the directory to be created
    generate_metadata_file()
    generate_embeddings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--data_dir', type=str, default='/home/asjindal/Work/tf/dependency_parsing_tf/dependency_parsing_tf/MNIST_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/home/asjindal/Work/tf/dependency_parsing_tf/dependency_parsing_tf/logs',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

