# tensorrt_carplate_rec
# purpose
This project is used to convert pytorch license plate recognition program into tensorrt
Because it is subject to tensorrt permute (shuffle) layer(single batch)
So far, the program can only be used for single picture recognition(single batch)
# usage
Clone this project to your directory
First, you need to create 'model/' folder to store the model 'crnn_best.pth'
Then, running trt.py for image recognition
