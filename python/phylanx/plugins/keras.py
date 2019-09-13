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

@Phylanx
def convolve_layer(input_data,num_filter,kernel_size):
    num_inputs=np.shape(input_data)[0]
    result_size=(np.shape(input_data)[1]-kernel_size[0]+1)*(np.shape(input_data)[2]-kernel_size[1]+1)
    result=np.zeros((result_size*num_filter))
    for j in range(num_filter):
        kernel=random(kernel_size)
        tmp=np.zeros(((np.shape(input_data)[1]-kernel_size[0]+1),(np.shape(input_data)[2]-kernel_size[1]+1)))
        for i in range(num_inputs):
            tmp=tmp+conv2d(input_data[0],kernel)
        result[j*result_size:(j+1)*result_size]=flatten(tmp)
    return result

@Phylanx
def activation_layer(input_data,activation):
    result=False
    if activation=='relu':
        result=relu(input_data)
    if activation=='elu':
        result=relu(input_data)
    if activation=='sigmoid':
        result=sigmoid(input_data)
    if activation=='tanh':
        result=tanh(input_data)
    if activation=='softmax':
        result=softmax(input_data)
    return result


@Phylanx
def my_simple_network():
    input_data=np.ones((15,5,5))
    first_layer=convolve_layer(input_data,4,[3,3])
    # second_layer=convolve_layer(first_layer,4,[3,3])
    last_layer=activation_layer(first_layer,'relu')
    return last_layer