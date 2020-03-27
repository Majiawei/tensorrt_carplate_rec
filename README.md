# tensorrt_carplate_rec
# purpose
This project is used to convert pytorch license plate recognition program into tensorrt
Because it is subject to tensorrt permute (shuffle) layer(single batch)
So far, the program can only be used for single picture recognition(single batch)
# usage
Clone this project to your directory
First, you need to create 'model/' folder to store the model 'crnn_best.pth'
Then, running trt.py for image recognition

trt_color.py add a color branch to recognize LP color, trt_color_multibatch.py is LPR trt code inferenced with a batch.

# pretrained model
Model is available on https://pan.baidu.com/s/1VQMZWlHPNgcVYADKOtS25w (pc0v)
