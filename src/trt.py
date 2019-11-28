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


def decode(pred):
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
    # for index in range(output_list.shape[0]):
    #     if output_list[index] != 75:
    #         if index == 0:
    #             pred_char += (dic[int(output_list[index])-1])
    #         else:
    #             if output_list[index] != output_list[index-1]:
    #                 pred_char += (dic[int(output_list[index])])

    return char_list

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 32, 100)
    OUTPUT_NAME1 = "fc1"
    OUTPUT_NAME2 = "fc2"
    OUTPUT_NAME3 = "cat"
    OUTPUT_SIZE = 76
    DTYPE = trt.float32
    RELU_alpha = 0.2


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


def populate_network(network, weights):
    # below is network define, no need to care
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(
        name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # Backbone Sequential
    conv0 = conv_trt(weights, input_tensor, 'cnn.conv0', network, 64)
    print('conv0')
    print(conv0.get_output(0).shape)
    bn0 = bn_trt(weights, conv0.get_output(0), 'cnn.batchnorm0', network)
    relu0 = network.add_activation(input=bn0.get_output(
        0), type=trt.ActivationType.LEAKY_RELU)
    relu0.alpha = ModelData.RELU_alpha
    pooling0 = max_pooling_trt(relu0.get_output(0), network)
    print('pooling0')
    print(pooling0.get_output(0).shape)

    conv1 = conv_trt(weights, pooling0.get_output(0),
                     'cnn.conv1', network, 128)
    print('conv1')
    print(conv1.get_output(0).shape)
    bn1 = bn_trt(weights, conv1.get_output(0), 'cnn.batchnorm1', network)
    relu1 = network.add_activation(input=bn1.get_output(
        0), type=trt.ActivationType.LEAKY_RELU)
    relu1.alpha = ModelData.RELU_alpha
    pooling1 = max_pooling_trt(relu1.get_output(0), network)
    print('pooling1')
    print(pooling1.get_output(0).shape)

    conv2 = conv_trt(weights, pooling1.get_output(0),
                     'cnn.conv2', network, 256)
    print('conv2')
    print(conv2.get_output(0).shape)
    bn2 = bn_trt(weights, conv2.get_output(0), 'cnn.batchnorm2', network)
    relu2 = network.add_activation(input=bn2.get_output(
        0), type=trt.ActivationType.LEAKY_RELU)
    relu2.alpha = ModelData.RELU_alpha
    conv3 = conv_trt(weights, relu2.get_output(0), 'cnn.conv3', network, 256)
    print('conv3')
    print(conv3.get_output(0).shape)
    bn3 = bn_trt(weights, conv3.get_output(0), 'cnn.batchnorm3', network)
    relu3 = network.add_activation(input=bn3.get_output(
        0), type=trt.ActivationType.LEAKY_RELU)
    relu3.alpha = ModelData.RELU_alpha
    pooling2 = max_pooling_trt(relu3.get_output(0), network, stride=(2, 1))
    print('pooling2')
    print(pooling2.get_output(0).shape)

    conv4 = conv_trt(weights, pooling2.get_output(0),
                     'cnn.conv4', network, 512)
    print('conv4')
    print(conv4.get_output(0).shape)
    bn4 = bn_trt(weights, conv4.get_output(0), 'cnn.batchnorm4', network)
    relu4 = network.add_activation(input=bn4.get_output(
        0), type=trt.ActivationType.LEAKY_RELU)
    relu4.alpha = ModelData.RELU_alpha
    conv5 = conv_trt(weights, relu4.get_output(0), 'cnn.conv5', network, 512)
    print('conv5')
    print(conv5.get_output(0).shape)
    bn5 = bn_trt(weights, conv5.get_output(0), 'cnn.batchnorm5', network)
    relu5 = network.add_activation(input=bn5.get_output(
        0), type=trt.ActivationType.LEAKY_RELU)
    relu5.alpha = ModelData.RELU_alpha
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

    network.mark_output(tensor=permute3.get_output(0))


def build_engine(weights):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = common.GiB(1)
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_cuda_engine(network)
# def load_random_test_case(pagelocked_buffer):
#     img = None
#     label = None
#     # generate a random num for img fetch
#     random_num = int(100 * random())
#     for _, (i, index, l) in enumerate(data_loader):
#         img = i
#         label = l
#         # use random num to break, this is a fast but bad implentation
#         if _ == random_num:
#             break
#     # reshape to remove 'batch' dim
#     img = img.reshape(1, 32, 100)
#     # flatten ndarray for input
#     img = img.numpy().ravel()
#     # Copy to the pagelocked input buffer
#     np.copyto(pagelocked_buffer, img)
#     return label
# Converts the input image to a CHW Numpy array
def normalize_image(cropped_image):
    imgH = 32
    imgW = 100
    mean = 0.588
    std = 0.193
    image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    image = cv2.resize(image, (0,0), fx=imgW/w, fy=imgH/h, interpolation=cv2.INTER_CUBIC)
    image = (np.reshape(image, (imgH, imgW, 1))).transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.
    # print(image)
    # print(type(image))
    image = (image - mean)/std
    print(image.shape)
    return image.astype(trt.nptype(ModelData.DTYPE)).ravel()
    # image = torch.from_numpy(image).type(torch.FloatTensor)
    # image.sub_(mean).div_(std)
    # if torch.cuda.is_available():
    #     image = image.cuda()
    # image = image.view(1, *image.size())
    # # Resize, antialias and transpose the image to CHW.
    # c, h, w = ModelData.INPUT_SHAPE
    # return np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
def load_normalized_test_case(test_image, pagelocked_buffer):
    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(cv2.imread(test_image)))
    return test_image

# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream
def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()
def main():
    test_image = "7.jpg"
    # use cpu for robust
    dev = torch.device('cpu')
    # custom path(edit needed)
    weights = torch.load('../model/crnn_best.pth', dev)
    # Do inference with TensorRT.
    with build_engine(weights) as engine:
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        # inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            test_case = load_normalized_test_case(test_image, h_input)
            # label = load_random_test_case(pagelocked_buffer=inputs[0].host)
            do_inference(context, h_input, d_input, h_output, d_output, stream)
            # print(h_output)
            # print(len(h_output))
            # [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            output = h_output.reshape([28, 76])
            # idx = np.argmax(output, axis=1)
            pred = decode(output)
            # print("Test Case label: " + str(label))
            print("Prediction: " + pred)


if __name__ == '__main__':
    main()
