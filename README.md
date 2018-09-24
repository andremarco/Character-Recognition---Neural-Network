# Character-Recognition---Neural-Network

Project made by:
* Andrea Marcocchia
* Matteo Manzari

## Data Generation
We generate 2592 images, obtained from all the possible combinations of 11 fonts and 96 characters (letters, numbers and special symbols). Not all fonts have the bold and the italics modality. In this first step the character image is centered and its quality is of *64x64* pixels. The images are in black and white, so the last channel has only one dimension.
The basic images are filled in with values in the range 0-255, where the 0’s are the background. The images are augmentated in the next step.

## Data Augmentation
In order to improve our dataset we use five different augmentation techniques:
• Rescale: we zoom the image using a random parameter to decide the rescale intensity
• Salt and pepper: we add the notorious “salt and pepper” technique
• Rotate: we rotate the image using a randomly chosen angle. The image could be rotate with
an angle in the range (-45,+45) grades, in order to avoid strange rotation of the image. The
background of the image is always filled in using white color
• Traslation: we traslate the image in two different directions (left/right and up/down) by
choosing a random parameter
• Random gaussian noise: we add a random number according to a gaussian distribution with
mean 0 and small standard deviation.

For each orginal image we create 100 augmentated copies. Each copy is augmented with a random
number of functions, casually chosen. At the end of the augmentation stage we have 259200 images.
The augmented images are filled in with values in the range 0-1 and have white background and black letters. The images are saved in a .npy file, and their labels are stored in a dictionary data- structure.

## Model
To build our model we used the functional API technique from Keras and we developed it progressively with sequential blocks.
Our input layer received in input an image of shape (64x64x1).
The first layer is a convolutional one, composed by 126 filters, with kernel size (8x8) and strides equal to 2. In this layer the parameter padding is setted as same, in order to obtain an output image of the same dimension of the input one.
The second layer is a Batch Normalization layer, and it is described later in the report.
The third layer is a Max Pooling, with size (2x2).
The final layer of the convolutional block is a Dropout with a low probability (5%).
At this stage we add another block with a similar structure of the previous one, characterized by a convolutional layer with 256 filters and 2x2 kernel size and two Max Pooling layers, both with size 2x2.
The last Max Pooling layer is followed by a flatten layer, in order to obtain a linear output, and a Dropout with probability thershold setted at 50%.
The droput layer is directly linked to the four dense output layers, with the folowing structure:
• output_char: 94 neurons
• output_font: 11 neurons
• output_bold: 1 neuron
• output_italics: 1 neuron

We train our algorithm with batch size equal to 3000 so that, in each epoch, the parameters are updated every 3000 obervations. We use 20 epochs, that is the maximum possible, in order to run the script in approx 900 seconds. We also use a validation testset, in order to optimize the hyperparameter.

In our model we use the Adam optimizer function, that is an extension of the stochastic gradient descent. We set the learning rate of the Adam optimizer equal to 0.002 (default value). We also tried to use other optimizers, such that SGD (specifying parameters as decay and momentum, trying to replicate the optimizer used in a famous and well-perfomer NN), Adagrad and Adadelta, but Adam always give us the better performance.

In defining and configuring the model we tried over 100 different architectures for our Neural Network, always experiencing some critical issues with overfitting. In order to avoid overfitting we applyied some functions such as:
• Dropout: at each training stage, individual nodes are either dropped out of the net with probability 1-p or kept with probability p. Usually the droput is used in the fully connected layer. Anyway, as Gal & Ghahramani (2016) wrote, it’s possible to avoid overfitting using dropout also in the convolutional layer, using small probability values.
• BatchNormalization: to reduce the amount by what the hidden unit values shift around. The resulting parameters are centered around zero with variance 1 (we maintain the default values for the parameters)
• L1 loss function: to add a regularization term in order to prevent the coefficients to fit so perfectly to overfit. The L1 loss function works on the sum of the weights.
• Random initialization for Kernel initializer: to increase the variability of the parameters (in particular the first iterations of the CNN). We use some random distributions, such random uniform and random normal.

## Test
Before applying the prediction function, we need to normalize the hidden test dataset. We use this technique in order to rescale the image values in the range 0-1.

The results of the prediction have to be analyzed in two different ways:
- font and char: the output of the model is a series of probability values, so that for each possible
class the result is a value in the range 0-1 and the sum of all the classes is equal to 1. This result is possible because of the softmax activation function, used in the output layer.
- Bold and italics: : the output of the model is a number between 0-1. If that value is bigger than 0.5, we classify it as 1, on the other hand as 0. In this case we use the sigmoid activation function for the output layer. The number 1 means that the selected image is bold or italics.

The final result for the complete test dataset is saved in the Partial_result.csv file.
The submitted model has a global accuracy of 0.4621224210646301, and its detailed accuracy for each area is:
• Char: 0.459592
• Font: 0.273111
• Bold: 0.650405
• Italics: 0.561154

This results show that the reached level obtained for the char is acceptable (the random probability to select the correct character is 0.01%), but we surealy could perform better for italics because the final accuracy is not far from a completely random classification. The font accuracy (0.273), even with more complex model applied, does not move to better values. Anyway this result is better than the 0.9 that could be obtained randomly pick a font. Probably font and italics are complex features to be modelled by the applied NN. We tried to fix this problem by using more complex models, and the performance registred are a very good level of accuracy in the validation dataset, but a maximum of 0.41 in the test. Assuming that this results are due to overfitting, we opted to apply a simpler model, in order to have a better control of the parameter behaviour.
The final CNN submitted model reachs these results in the validation set:
• val_output_font_acc: 0.0938
• val_output_char_acc: 0.7203
• val_output_bold_acc: 0.6665
• val_output_italics_acc: 0.6738
Extra Points
In order to complete the Extra-Points, we created a new model with intermediate outputs and the same weights of the trained model, in order to deeply understand the behaviour of the hidden layers. The intermediate outputs are saved in .png format, and the code used to save the images is stored in the main.py file.
In the image_intermediate_1.png it’s possible to visualize the behaviour of the first convolutional layer. We report 128 images (one for each filter) and each filter tries to model a different features of the image.
Regarding image_intermediate_2.png it’s possible to understand how the output layer decides how to classify the font for the selected image.
