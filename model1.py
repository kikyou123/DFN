from op import *
import tensorflow as tf
import os
import numpy as np

def add_upscale(input):
    prev_shape = input.get_shape()
    size = [2 * int(s) for s in prev_shape[1:3]]
    return tf.image.resize_nearest_neighbor(input, size)

def dfl(input, filters, ker = 9):
    '''

    :param input: input image of size [batch_size, h, w, 3]
    :param filter: filter [batch_size, h, w, 81]
    :return: output image of size [batch_size, h, w, 3]
    '''
   # filter_size = ker * ker
   # expand_filter = np.reshape(np.eye(filter_size, filter_size), (filter_size, 1, ker, ker))
   # expand_filter = np.transpose(expand_filter, (2,3,1,0))
   # expand_input = tf.nn.conv2d(input, expand_filter, strides = [1,1,1,1], padding = 'SAME')

   # output = expand_input * filter
   # output = tf.reduce_sum(output, axis = -1, keep_dims = True)
    image_patches = tf.extract_image_patches(input, [1, ker, ker, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='SAME')
    output = tf.reduce_sum(tf.multiply(image_patches, filters), 3, keep_dims=True)
    return output
    
def pred(input, hidden, filter = 9):
    # encoder
    h1 = lrelu(conv2d(input, 32, d_h = 1, d_w = 1, name = 'h1'))
    h2 = lrelu(conv2d(h1, 32, d_h = 2, d_w = 2, name = 'h2'))
    h3 = lrelu(conv2d(h2, 64, d_h = 1, d_w = 1, name = 'h3'))
    h4 = lrelu(conv2d(h3, 64, d_h = 1, d_w = 1, name = 'h4'))
    
    # middle
    h5 = lrelu(conv2d(h4, 128, d_h = 1, d_w = 1, name = 'h5'))
    hidden1 = lrelu(conv2d(hidden, 128, d_h = 1, d_w = 1, name = 'hidden_1'))
    hidden2 = lrelu(conv2d(hidden1, 128, d_h = 1, d_w = 1, name = 'hidden_2'))
    h6 = tf.add(hidden2, h5)
    hidden_state = h6

    #decoder
    h7 = lrelu(conv2d(h6, 64, d_h = 1, d_w = 1, name = 'h7'))
    h8 = lrelu(conv2d(h7, 64, d_h = 1, d_w = 1, name = 'h8'))
    h9 = add_upscale(h8)
    h10 = lrelu(conv2d(h9, 64, d_h = 1, d_w = 1, name = 'h10'))
    h11 = lrelu(conv2d(h10, 64, d_h = 1, d_w = 1, name = 'h11'))
    h12 = lrelu(conv2d(h11, 128, k_h=1, k_w=1, d_h = 1, d_w = 1, name = 'h12'))
    l_filter = conv2d(h12, filter * filter, k_h=1, k_w=1, d_h = 1, d_w = 1, name = 'h13')
    l_filter = tf.nn.softmax(l_filter)
    #filter
    output = dfl(input[:,:,:,-1:], l_filter, filter)
    return output, hidden_state

def model(inputs, input_seqlen = 3, target_seqlen = 3, buffer_len = 1, filter = 9, reuse = False):
    # inputs : [batch_size, seqlen, image_size, image_size, n_channel]
    # return : [batch_size, seqlen, image_size, image_size, n_channel]
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        network_template = tf.make_template('pred', pred)
        batch_size, seqlen, image_size, image_size, n_channel = inputs.get_shape().as_list()
        inputs = tf.transpose(inputs, (0,2,3,1,4))
        inputs = tf.reshape(inputs, (batch_size, image_size, image_size, seqlen*n_channel))
        hidden = tf.zeros((batch_size, image_size // 2, image_size // 2, 128))
        outputs = []
        
        for i in range(input_seqlen - buffer_len + target_seqlen):
            pred_input = inputs[..., 0: buffer_len]
            output, hidden = network_template(pred_input, hidden, filter)
            inputs = inputs[..., 1:None]

            if i >= input_seqlen - buffer_len:
                outputs.append(output)
                if inputs.get_shape()[-1] == 0:
                    inputs = output
                else:
                    inputs = tf.concat([inputs, output], axis = 1)
        outputs = tf.stack(outputs, axis = 0)
        outputs = tf.transpose(outputs, (1,0,2,3,4))

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'gen')
    return outputs, vars

def get_loss(targets, outputs, target_seqlen):
    outputs = tf.clip_by_value(outputs, np.finfo(np.float32).eps, 1 - np.finfo(np.float32).eps)
    loss = -targets * tf.log(outputs) - (1 - targets) * tf.log(1 - outputs)
    loss = tf.reduce_mean(loss)
    #loss = tf.reduce_mean(tf.square(targets - outputs))
    #loss = tf.reduce_mean(tf.abs(targets - outputs))
    return loss

def create_optimizers(loss, params, learning_rate):
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list = params)
    return opt

