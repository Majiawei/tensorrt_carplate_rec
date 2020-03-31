import sys

import numpy as np
import tensorrt as trt
import torch
import cv2
from random import random

import common as common
import pycuda.driver as cuda
import pycuda.autoinit
import time

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
    BATCH_SIZE = 12
    INPUT_SHAPE = (1, 3, 32, 100)
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
    #定义网络输入
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
    permute1.first_transpose = trt.Permutation([3, 0, 1, 2])
    permute1.reshape_dims = [permute1.get_output(0).shape[0], permute1.get_output(0).shape[1], 1, 1, -1]
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
    permute2.first_transpose = trt.Permutation([3, 0, 1, 2])
    permute2.reshape_dims = [permute2.get_output(0).shape[0], permute2.get_output(0).shape[1], 1, 1, -1]
    print('permute2')
    print(permute2.get_output(0).shape)

    fc2 = fc_trt(weights, permute2.get_output(
        0), 'fc2', network, ModelData.OUTPUT_SIZE)
    fc2.get_output(0).name = ModelData.OUTPUT_NAME2
    print('fc2')
    print(fc2.get_output(0).shape)

    cat1 = network.add_concatenation([fc1.get_output(0), fc2.get_output(0)])
    cat1.axis = 0
    cat1.get_output(0).name = ModelData.OUTPUT_NAME3
    permute3 = network.add_shuffle(cat1.get_output(0))
    permute3.reshape_dims = [0, 0, -1]
    permute3.second_transpose = trt.Permutation([1,2,0])
    print('permute3')
    print(permute3.get_output(0).shape)
    re_softmax = network.add_softmax(input=permute3.get_output(0))
    re_softmax.axes = 2
    print('re_softmax')
    print(re_softmax.get_output(0).shape)
    permute4 = network.add_shuffle(re_softmax.get_output(0))
    permute4.first_transpose = trt.Permutation([0,2,1])
    print('permute4')
    print(permute4.get_output(0).shape)

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

    #定义网络输出
    network.mark_output(tensor=permute4.get_output(0))
    network.mark_output(tensor=fc_c.get_output(0))


def build_engine(weights, weights_color, batch_size=4):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = common.GiB(1)
        builder.max_batch_size = batch_size
        #定义网络结构，并将赋值训练好的weights
        populate_network(network, weights, weights_color)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


def loadImage(test_image_txt):
    f_r = open(test_image_txt,'r')
    lines = f_r.readlines()
    images=[]
    labels=[]
    for line in lines:
        label = line.split(' ')[1]

        img_path = line.split(' ')[0]
        image = cv2.imread(img_path)
        imgH = 32
        imgW = 100
        h, w, c = image.shape
        image = cv2.resize(image, (0,0), fx=imgW/w, fy=imgH/h, interpolation=cv2.INTER_CUBIC)
        image = (np.reshape(image, (imgH, imgW, 3))).transpose(2, 0, 1)
        image = image.astype(np.float32) / 255.
        images.append(image)
        labels.append(label)

    return images, labels

def main():
    #加载所有测试图像
    test_image_txt = '/rdata/qi.liu/code/LPR/ezai/all_projects/carplate_recognition/data/test_new/line1_10.txt'
    images, labels = loadImage(test_image_txt)
    color_dict=['蓝','黄','绿','黑','白']
    #定义测试性能计数器
    count_text_right = 0
    count_color_right = 0
    # use cpu for robust
    dev = torch.device('cpu')

    #load trained weights
    weights = torch.load('../model/model_LPR_text.pth', dev)
    weights_color = torch.load('../model/model_LPR_color.pth', dev)


    #build engine
    engine = build_engine(weights, weights_color, ModelData.BATCH_SIZE)
    # allocate_buffers, 为输入输出分配空间，h_代表的是在GPU运算的变量，d_代表的是在CPU运算的变量
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    #create_context
    with engine.create_execution_context() as context:
        #对输入赋值
        inputs[0].host = np.array(images) #
        # inputs[0].host = images
        print(inputs[0].host.shape)

        start = time.time()
        #执行inference
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=ModelData.BATCH_SIZE)
        # trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        end = time.time()

        #对输出进行后处理，每个output都被flatten到一个一维向量，要先reshape
        text_preds = trt_outputs[0].reshape([-1, 28, 76])
        color_preds = trt_outputs[1].reshape([-1, 5])
        print(text_preds.shape, color_preds.shape)

        #解码输出结果，统计识别性能
        '''
        # results
            pred_txt: 车牌号
            probs :置信度
            pred_color：预测颜色
        '''
        for i in range(text_preds.shape[0]):
            output_text = text_preds[i,:,:]

            pred_txt, probs = decode(output_text)
            output_c = color_preds[i,:]
            max_index = np.argmax(output_c, 0)
            # print(output_text, output_c)
            pred_color = color_dict[max_index]
            # if pred_txt == labels[i]:
            #     count_text_right += 1
            # if pred_color == '蓝':
            #     count_color_right += 1
            print("color:"+pred_color+"  Prediction: " + pred_txt+"  Text_probs: " + str(probs))

    
    # print('Text Accuracy: ', count_text_right/len(labels), str(count_text_right), str(len(labels)))
    # print('Color Accuracy: ', count_color_right/len(labels),str(count_color_right), str(len(labels)))
    print(end-start)


if __name__ == '__main__':
    main()
