import sys

import numpy as np
import tensorrt as trt
import torch
import cv2
from random import random

import common
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def decode_text(pred):
    dic = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '京',
           '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '领', '学', '使', '警', "挂", '港', '澳', "电", "*"]

    output_list = pred.argmax(axis=1)
    # print(output_list)
    # pred_char = ''
    char_list = []
    for i in range(len(output_list)):
        if output_list[i] != 0 and (not (i > 0 and output_list[i - 1] == output_list[i])):
            char_list.append(dic[output_list[i] - 1])
    return ''.join(char_list)

def decode(pred):
    dic = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '京',
           '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '领', '学', '使', '警', "挂", '港', '澳', "电", "*"]
    probs_list=[]
    max_probs = 0
    print(pred.shape)
    output_list = pred.argmax(axis=1)
    char_list = []
    for i in range(len(output_list)):
        # print(str(i), output_list[i], pred[i][output_list[i]])
        if output_list[i] !=0:
            if i == 0:
                char_list.append(dic[output_list[i] - 1])
                max_probs = pred[i][output_list[i]]

            else:
                if not output_list[i - 1] == output_list[i]:
                    char_list.append(dic[output_list[i] - 1])
                    probs = pred[i][output_list[i]]
                    if max_probs != 0:
                        probs_list.append(max_probs)
                    max_probs = probs

                else:
                    probs = pred[i][output_list[i]]
                    if probs > max_probs:
                        max_probs = probs
    probs_list.append(max_probs)
        
    return ''.join(char_list), probs_list

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (3, 32, 100)
    OUTPUT_NAME1 = "fc1"
    OUTPUT_NAME2 = "fc2"
    OUTPUT_NAME3 = "cat"
    OUTPUT_COLOR_NAME = "fc_c"

    OUTPUT_SIZE = 76
    OUTPUT_COLOR_SIZE = 5
    DTYPE = trt.float32


def bn_trt(weights, prev_output, layer_name, network):
    """
    exec bn operation user eltwise in tensorrt
    :param network: tensorrt network
    :param weights: model pretrained params -> OrderedDict
    :param prev_output: prev layer output -> tensorrt tensor
    :param layer_name: layer's name for get bn params refer to pretrained model -> str
    :return: bn's output -> tensorrt tensor
    """
    # # get bn params -> torch.
    # weight = weights[layer_name + ".weight"].numpy()
    # # weight = np.expand_dims(weight, -1)
    # # weight = np.expand_dims(weight, -1)
    # bias = weights[layer_name + ".bias"].numpy()
    # # bias = np.expand_dims(bias, -1)
    # # bias = np.expand_dims(bias, -1)
    # mean = weights[layer_name + ".running_mean"].numpy()
    # # mean = np.expand_dims(mean, -1)
    # # mean = np.expand_dims(mean, -1)
    # var = weights[layer_name + ".running_var"].numpy()
    # var = np.expand_dims(var, -1)
    # var = np.expand_dims(var, -1)
    # # calculate weight / sqrt(var - epsilon)
    # scale = weight / np.sqrt(var + sys.float_info.epsilon)
    # # calculate bn by eltwise
    # trt_scale = network.add_constant(scale.shape, scale)
    # trt_mean = network.add_constant(mean.shape, mean)
    # trt_bias = network.add_constant(bias.shape, bias)
    # minus_layer = network.add_elementwise(prev_output,
    #                                       trt_mean.get_output(0),
    #                                       trt.ElementWiseOperation.SUB)
    # multi_layer = network.add_elementwise(minus_layer.get_output(0),
    #                                       trt_scale.get_output(0),
    #                                       trt.ElementWiseOperation.PROD)
    # add_layer = network.add_elementwise(multi_layer.get_output(0),
    #                                     trt_bias.get_output(0),
    #                                     trt.ElementWiseOperation.SUM)
    scale = weights[layer_name + ".weight"].detach().cpu().numpy() / np.sqrt(weights[layer_name + ".running_var"].detach().cpu().numpy() + sys.float_info.epsilon)
    bias = weights[layer_name + ".bias"].detach().cpu().numpy() - weights[layer_name + ".running_mean"].detach().cpu().numpy() * scale
    power = np.ones_like(scale)

    layer = network.add_scale(prev_output, trt.ScaleMode.CHANNEL, bias, scale, power)

    return layer


def conv_trt(weights, prev_output, layer_name, network,
             num_out: int, kernel: tuple = (3, 3), stride: tuple = (1, 1), padding: tuple = (1, 1)):
    # get params
    weight = weights[layer_name + '.weight'].numpy()
    bias = weights[layer_name + '.bias'].numpy()
    # add in network
    conv = network.add_convolution(input=prev_output,
                                   num_output_maps=num_out,
                                   kernel_shape=kernel,
                                   kernel=weight,
                                   bias=bias
                                   )
    conv.stride = stride
    conv.padding = padding
    return conv


def max_pooling_trt(prev_output, network,
                    kernel: tuple = (2, 2), stride: tuple = (2, 2)):
    max_pooling = network.add_pooling(input=prev_output,
                                      type=trt.PoolingType.MAX,
                                      window_size=kernel)
    max_pooling.stride = stride
    return max_pooling


def fc_trt(weights, prev_output, layer_name, network, num_outputs):
    # get params
    weight = weights[layer_name + '.weight'].numpy()
    bias = weights[layer_name + '.bias'].numpy()
    # add fc in network
    fc = network.add_fully_connected(input=prev_output,
                                     num_outputs=num_outputs,
                                     kernel=weight,
                                     bias=bias
                                     )
    return fc


def populate_network(network, weights, weights_color):
    # below is network define, no need to care
    # Configure the network layers based on the weights provided.
    # mark input
    input_tensor = network.add_input(
        name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # Backbone Sequential
    conv0 = conv_trt(weights, input_tensor, 'cnn_1.conv0', network, 64)
    print('conv0')
    print(conv0.get_output(0).shape)
    bn0 = bn_trt(weights, conv0.get_output(0), 'cnn_1.batchnorm0', network)
    relu0 = network.add_activation(input=bn0.get_output(
        0), type=trt.ActivationType.RELU)
    pooling0 = max_pooling_trt(relu0.get_output(0), network)
    print('pooling0')
    print(pooling0.get_output(0).shape)

    conv1 = conv_trt(weights, pooling0.get_output(0),
                     'cnn_1.conv1', network, 128)
    print('conv1')
    print(conv1.get_output(0).shape)
    bn1 = bn_trt(weights, conv1.get_output(0), 'cnn_1.batchnorm1', network)
    relu1 = network.add_activation(input=bn1.get_output(
        0), type=trt.ActivationType.RELU)
    pooling1 = max_pooling_trt(relu1.get_output(0), network)
    print('pooling1')
    print(pooling1.get_output(0).shape)

    conv2 = conv_trt(weights, pooling1.get_output(0),
                     'cnn_1.conv2', network, 256)
    print('conv2')
    print(conv2.get_output(0).shape)
    bn2 = bn_trt(weights, conv2.get_output(0), 'cnn_1.batchnorm2', network)
    relu2 = network.add_activation(input=bn2.get_output(
        0), type=trt.ActivationType.RELU)
    # Backbone Sequential-2
    conv3 = conv_trt(weights, relu2.get_output(0), 'cnn_2.conv3', network, 256)
    print('conv3')
    print(conv3.get_output(0).shape)
    bn3 = bn_trt(weights, conv3.get_output(0), 'cnn_2.batchnorm3', network)
    relu3 = network.add_activation(input=bn3.get_output(
        0), type=trt.ActivationType.RELU)
    pooling2 = max_pooling_trt(relu3.get_output(0), network, stride=(2, 1))
    print('pooling2')
    print(pooling2.get_output(0).shape)

    conv4 = conv_trt(weights, pooling2.get_output(0),
                     'cnn_2.conv4', network, 512)
    print('conv4')
    print(conv4.get_output(0).shape)
    bn4 = bn_trt(weights, conv4.get_output(0), 'cnn_2.batchnorm4', network)
    relu4 = network.add_activation(input=bn4.get_output(
        0), type=trt.ActivationType.RELU)
    conv5 = conv_trt(weights, relu4.get_output(0), 'cnn_2.conv5', network, 512)
    print('conv5')
    print(conv5.get_output(0).shape)
    bn5 = bn_trt(weights, conv5.get_output(0), 'cnn_2.batchnorm5', network)
    relu5 = network.add_activation(input=bn5.get_output(
        0), type=trt.ActivationType.RELU)
    # Branch 1 Sequential
    pooling7 = max_pooling_trt(relu5.get_output(0), network)
    print('pooling7')
    print(pooling7.get_output(0).shape)

    conv9 = conv_trt(weights, pooling7.get_output(0), 'branch1.conv9', network,
                     512, kernel=(2, 2), stride=(2, 2), padding=(0, 0))
    print('conv9')
    print(conv9.get_output(0).shape)
    bn9 = bn_trt(weights, conv9.get_output(0), 'branch1.batchnorm9', network)
    relu9 = network.add_activation(
        input=bn9.get_output(0), type=trt.ActivationType.RELU)
    permute1 = network.add_shuffle(relu9.get_output(0))
    permute1.first_transpose = trt.Permutation([2, 0, 1])
    permute1.reshape_dims = [0, 1, 1, -1]
    print('permute1')
    print(permute1.get_output(0).shape)

    fc1 = fc_trt(weights, permute1.get_output(
        0), 'fc1', network, ModelData.OUTPUT_SIZE)
    fc1.get_output(0).name = ModelData.OUTPUT_NAME1
    print('fc1')
    print(fc1.get_output(0).shape)

    # Branch 2 Sequential
    pooling3 = max_pooling_trt(relu5.get_output(0), network, stride=(2, 1))
    conv7 = conv_trt(weights, pooling3.get_output(0),
                     'branch2.conv7', network, 512)
    print('conv7')
    print(conv7.get_output(0).shape)
    bn7 = bn_trt(weights, conv7.get_output(0), 'branch2.batchnorm7', network)
    relu7 = network.add_activation(
        input=bn7.get_output(0), type=trt.ActivationType.RELU)

    conv8 = conv_trt(weights, relu7.get_output(0), 'branch2.conv8', network,
                     512, kernel=(2, 2), stride=(1, 1), padding=(0, 0))
    print('conv8')
    print(conv8.get_output(0).shape)
    bn8 = bn_trt(weights, conv8.get_output(0), 'branch2.batchnorm8', network)
    relu8 = network.add_activation(
        input=bn8.get_output(0), type=trt.ActivationType.RELU)

    permute2 = network.add_shuffle(relu8.get_output(0))
    permute2.first_transpose = trt.Permutation([2, 0, 1])
    permute2.reshape_dims = [0, 1, 1, -1]
    fc2 = fc_trt(weights, permute2.get_output(
        0), 'fc2', network, ModelData.OUTPUT_SIZE)
    fc2.get_output(0).name = ModelData.OUTPUT_NAME2
    print('fc2')
    print(fc2.get_output(0).shape)

    cat1 = network.add_concatenation([fc1.get_output(0), fc2.get_output(0)])
    cat1.axis = 0
    cat1.get_output(0).name = ModelData.OUTPUT_NAME3
    permute3 = network.add_shuffle(cat1.get_output(0))
    permute3.reshape_dims = [0, -1]
    print('permute3')
    print(permute3.get_output(0).shape)
    # permute3.reshape_dims = [0, -1]
    re_softmax = network.add_softmax(input=permute3.get_output(0))
    re_softmax.axes = 2
    print('re_softmax')
    print(re_softmax.get_output(0).shape)

    #branch color
    c_pooling0 = max_pooling_trt(relu2.get_output(0), network, stride=(2, 2))
    c_conv0 = conv_trt(weights_color, c_pooling0.get_output(0),
                     'color_branch.c_conv0', network, 128)
    print('c_conv0')
    print(c_conv0.get_output(0).shape)
    c_relu0 = network.add_activation(
        input=c_conv0.get_output(0), type=trt.ActivationType.RELU)
    
    c_pooling1 = max_pooling_trt(c_relu0.get_output(0), network, stride=(2, 2))

    c_conv1 = conv_trt(weights_color, c_pooling1.get_output(0), 'color_branch.c_conv1', network, 64)
    print('c_conv1')
    print(c_conv1.get_output(0).shape)
    c_relu1 = network.add_activation(
        input=c_conv1.get_output(0), type=trt.ActivationType.RELU)

    fc_c = fc_trt(weights_color, c_relu1.get_output(
        0), 'fc_c', network, ModelData.OUTPUT_COLOR_SIZE)
    fc_c.get_output(0).name = ModelData.OUTPUT_COLOR_NAME
    print('fc_c')
    print(fc_c.get_output(0).shape)
    c_softmax = network.add_softmax(input=fc_c.get_output(0))
    c_softmax.axes = 1
    print('c_softmax')
    print(c_softmax.get_output(0).shape)

    # mark output
    network.mark_output(tensor=re_softmax.get_output(0))
    # network.mark_output(tensor=permute3.get_output(0))

    network.mark_output(tensor=c_softmax.get_output(0))


def build_engine(weights, weights_color):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = common.GiB(1)
        populate_network(network, weights, weights_color)
        # Build and return an engine.
        return builder.build_cuda_engine(network)

def normalize_image(image):
    imgH = 32
    imgW = 100
    h, w, c = image.shape
    image = cv2.resize(image, (0,0), fx=imgW/w, fy=imgH/h, interpolation=cv2.INTER_CUBIC)
    image = (np.reshape(image, (imgH, imgW, 3))).transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.

    print(image.shape)
    return image.astype(trt.nptype(ModelData.DTYPE)).ravel()

def load_normalized_test_case(test_image, pagelocked_buffer):
    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(cv2.imread(test_image)))
    return test_image

# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    h_output_c = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(2)), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    d_output_c = cuda.mem_alloc(h_output_c.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output,h_output_c, d_output_c, stream 

def do_inference(context, h_input, d_input, h_output, d_output, h_output_c, d_output_c, stream ):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output), int(d_output_c)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    cuda.memcpy_dtoh_async(h_output_c, d_output_c, stream)

    # Synchronize the stream
    stream.synchronize()
def main():
    #test_image loading
    test_image = "/rdata/qi.liu/code/TRT/tensorrt_carplate_rec/src/7.jpg"
    # test_image = "/rdata/qi.liu/code/TRT/tensorrt_carplate_rec/src/carplate_test.jpg"

    color_dict=['蓝','黄','绿','黑','白']
    # use cpu for robust
    dev = torch.device('cpu')

    #load trained weights
    weights = torch.load('../model/model_LPR_text.pth', dev)
    weights_color = torch.load('../model/model_LPR_color.pth', dev)
    

    #build engine
    engine = build_engine(weights, weights_color)
    #allocate_buffers
    h_input, d_input, h_output, d_output,h_output_c, d_output_c, stream = allocate_buffers(engine)

    #create context
    with engine.create_execution_context() as context:
        # synchronize input data(cpu) to gpu
        test_case = load_normalized_test_case(test_image, h_input)
        # conduct inference
        do_inference(context, h_input, d_input, h_output, d_output, h_output_c, d_output_c, stream )
        # output is a 1D tensor, first reshape
        # print(h_output.shape, h_output_c.shape)
        output_text = h_output.reshape([28, 76])

        '''
        # results
            pred_txt: 车牌号
            probs :置信度
            pred_color：预测颜色
            color_confi: 颜色置信度
        '''
        pred_txt, probs = decode(output_text)
        # print(pred_txt, probs)
        # print(decode_text(output_text))
        # print(h_output_c)
        max_index = np.argmax(h_output_c)
        pred_color = color_dict[max_index]
        color_confi = h_output_c[max_index]
        print("color:"+pred_color+"  color_confi: "+str(color_confi)+"  Prediction: " + pred_txt+"  Text_probs: " + str(probs))



if __name__ == '__main__':
    main()
