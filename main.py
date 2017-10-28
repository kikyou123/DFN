import tensorflow as tf
import os
import numpy as np
import importlib
import random
import datetime
from model3 import *
from dataset_mnist import  *
from util import  *
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

filters = 9
max_iter = 100 * 200
decay_after = 20 * 200
batch_size = 16
hidth = 64
width = 64
input_seqlen = 10
target_seqlen = 10
buffer_len = 1
random_seed = 123
lr = 0.001
print lr

out = open('log1.txt', 'w')
checkpoint_dir = 'checkpoint1'
train_dir = 'train1'
test_dir = 'test1'
gt_dir = 'gt1'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(gt_dir):
    os.makedirs(gt_dir)

def setup_tensorflow():
    config = tf.ConfigProto()
    sess = tf.Session(config = config)
    with sess.graph.as_default():
        tf.set_random_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    return sess

def next_batch(dh):
    track = dh.GetBatch()
    track = np.transpose(track, [0, 4, 2, 3, 1])
    batch_input = track[:, : input_seqlen]
    batch_target = track[:, input_seqlen:]
    return batch_input, batch_target

def _train(lr):
    #lr = 0.001
    input_track = tf.placeholder(tf.float32, [batch_size, input_seqlen, hidth, width, 1 ])
    target_track = tf.placeholder(tf.float32, [batch_size, target_seqlen, hidth, width, 1])
    learning_rate = tf.placeholder(tf.float32, [])
    gen_track, vars = model(input_track, input_seqlen, target_seqlen, buffer_len, filters)
    loss = get_loss(target_track, gen_track, target_seqlen)
    opt = create_optimizers(loss, vars, lr)

    sess = setup_tensorflow()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #if 0:
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print 'train from %s' % (ckpt.model_checkpoint_path)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        print "train from scratch..."


    print('prepare data...')
    dataset = importlib.import_module('dataset_mnist')
    dh = dataset.DataHandle(num_frames = input_seqlen + target_seqlen)
    sample_input_track, sample_target_track = next_batch(dh)
    imsave1(sample_input_track, sample_target_track, 0, test_dir)
    sample_gen_track = sess.run(gen_track, {input_track: sample_input_track})
    imsave1(sample_input_track, sample_gen_track, 1, test_dir)
    #print "gen_track"
    #print sample_gen_track
    #gen_filter = sess.run(l_filter, {input_track: sample_input_track})
    #for i in range(target_seqlen):
    #    print "cnt", i
    #    print "gen_filter"
    #    print gen_filter[i]

    print("begin train...")
   # lr = 0.001
    for i in range(max_iter):
        batch = i + 1
        batch_input_track, batch_target_track = next_batch(dh)
        Loss, _ = sess.run([loss, opt], feed_dict = {input_track: batch_input_track, target_track: batch_target_track, learning_rate: lr})
        #Loss, _ = sess.run([loss, opt], feed_dict = {input_track: batch_input_track, target_track: batch_target_track})
        str = 'train batch[%d]%s, loss[%3.3f]' % (batch, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), Loss)
        print str
        out.write(str + '\n')

        if batch % 200 == 1:
            imsave1(batch_input_track, batch_target_track, batch, gt_dir)
            batch_gen_track = sess.run(gen_track, {input_track: batch_input_track})
            imsave1(batch_input_track, batch_gen_track, batch, train_dir)

        if batch % 200 == 1:
            test_loss, sample_gen_track = sess.run([loss, gen_track], {input_track: sample_input_track, target_track: sample_target_track})
            str = 'test batch[%d] %s, loss[%3.3f]' % (batch, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), test_loss)
            print str
            out.write(str + '\n')
            imsave1(sample_input_track, sample_gen_track, batch, test_dir)

        if batch % 2000 == 0:
            save(checkpoint_dir, saver, sess, batch)

        if batch % decay_after == 0:
            lr = lr * 0.5

def main(argv = None):
    print lr
    _train(lr)
if __name__ == '__main__':
    tf.app.run()

