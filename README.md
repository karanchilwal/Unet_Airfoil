# Unet_Airfoil
#INPUT:
The input to the model is 3*256*256 (C*H*W) image (shows the shape of the airfoil and the angle of attack) and an extra parameter reduced frequency( one more reynold's number).
Input image is converted from (3*256*256) to (512,1) with the help of Convolutional layers and batchnorm. Then reduced frequency parameter is concatenated to 512 features of input image making it (513,1) and further linear layers are used to train those input image features with the reduced frequency parameter and thus making it again (512, 1) features.
OUTPUT:
The output of the model is the pressure contour around the airfoil in the form of an image of 3*256*256 size.
The 512 features obtained from the input at the encoder part and the linear layer are then Deconvoluted to form image again. Thus giving us 3*256*256 image.
LOSS and optimizer:
Mean squared error loss is used and ADAM optimizer is used.
