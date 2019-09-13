#  Copyright (c) 2019 R. Tohid
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
from phylanx import Phylanx


@Phylanx
def in_top_k(predictions, targets, k):
    top_k = np.argsort(-predictions)[:, :k]
    target = reshape(targets, [-1, 1])  # noqa
    return np.any(target == top_k, -1)

# @Phylanx
# def convolve_layer(input_data,num_filter,kernel_size):
#     num_inputs=np.shape(input_data)[0]
#     result_size_row=np.shape(input_data)[1]-kernel_size[0]+1
#     result_size_col=np.shape(input_data)[2]-kernel_size[1]+1
#
#     result=np.zeros((num_filter,result_size_row,result_size_col))
#     for j in range(num_filter):
#         kernel=random(kernel_size)
#         tmp=np.zeros((result_size_row,result_size_col))
#         for i in range(num_inputs):
#             tmp=tmp+conv2d(input_data[i],kernel)
#         result[j]=tmp
#     return result


@Phylanx
def variable(initial_value):
    return initial_value

@Phylanx
def truncated_normal(input_shape, mean=0.0, stddev=1.0, seed=0):
    set_seed(seed)
    if len(input_shape)==4:
        flattened_shape=1
        for i in range(len(input_shape)):
            flattened_shape=flattened_shape*input_shape[i]
        return [random(flattened_shape, list("normal", mean, stddev)),input_shape]
    else:
        return [random(input_shape, list("normal", mean, stddev)), None]

@Phylanx
def convolve_layer(input_data,num_filter,kernel_size):
    num_inputs=np.shape(input_data)[0]
    result_size_row=np.shape(input_data)[1]-kernel_size[0]+1
    result_size_col=np.shape(input_data)[2]-kernel_size[1]+1
    result_size=result_size_row*result_size_col
    result=np.zeros((result_size*num_filter))
    for j in range(num_filter):
        kernel=random(kernel_size)
        tmp=np.zeros(((np.shape(input_data)[1]-kernel_size[0]+1),(np.shape(input_data)[2]-kernel_size[1]+1)))
        for i in range(num_inputs):
            tmp=tmp+conv2d(input_data[0],kernel)
        result[j*result_size:(j+1)*result_size]=flatten(tmp)

    output_shape=[num_filter,result_size_row,result_size_col]
    return result,output_shape


@Phylanx
def activation_layer(input_data,activation):
    result=False
    if activation=='relu':
        return relu(input_data)
    elif activation=='elu':
        return relu(input_data)
    elif activation=='sigmoid':
        return sigmoid(input_data)
    elif activation=='tanh':
        return tanh(input_data)
    elif activation=='softmax':
        return softmax(input_data)
    else:
        return 0


@Phylanx
def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                       num_filters]

    # initialise weights and bias for the filter
    weights = variable(truncated_normal(conv_filt_shape, 0.0,0.03,0))
    bias = variable(truncated_normal([num_filters]))

    # # setup the convolutional layer operation
    # out_layer = conv2d(input_data, weights)
    #
    # # add the bias
    # out_layer += bias
    #
    # # apply a ReLU non-linear activation
    # out_layer = tf.nn.relu(out_layer)
    #
    # # now perform max pooling
    # ksize = [1, pool_shape[0], pool_shape[1], 1]
    # strides = [1, 2, 2, 1]
    # out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
    #                            padding='SAME')

    return weights

@Phylanx
def my_simple_network():
    input_data=np.ones((15,5,5))
    first_layer=convolve_layer(input_data,4,[3,3])
    # second_layer=convolve_layer(first_layer,4,[3,3])
    last_layer=activation_layer(first_layer[0],'relu')
    return last_layer