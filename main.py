from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from keras.initializers import Constant
from keras import regularizers
import numpy as np
from contextlib import redirect_stdout
import h5py
from keras.models import model_from_json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras.backend as K
from keras.initializers import RandomNormal
from PIL import Image


# Load the train dataset, made by the augmented data
letter_aug= np.load('../data/data_augmented.npy')

# Set same parameters
b1=0.000017
lpar=0.030
r_norm = RandomNormal(mean=0.0, stddev=0.01, seed=None)  

# Load the label for the train dataset
label= np.load('../data/label_augmented.npy').item()


# Define the input for the CNN
visible_0= Input(shape=(64,64,1), name="input_ll")

# Ground Floor
conv0= Conv2D(filters=128, kernel_size=(8,8),strides=2, padding='same', name="convol1", activation = 'relu', kernel_initializer='random_uniform', bias_initializer=Constant(value=b1),kernel_regularizer=regularizers.l1(0.001))(visible_0) 
batch0 = BatchNormalization()(conv0)
pool0 = MaxPooling2D(pool_size=(2,2),strides=None, name="maxpoo1",)(batch0)
visible = Dropout(0.05)(pool0)

# 1 Floor

conv0_1= Conv2D(filters=256, kernel_size=(3,3),strides=2, padding='same', name="convol2", activation = 'relu', kernel_initializer='random_normal',bias_initializer=Constant(value=b1),kernel_regularizer=regularizers.l1(0.001))(visible) 
batch1 = BatchNormalization()(conv0_1)
pool0_1 = MaxPooling2D(pool_size=(2,2),strides=None)(batch1)
drop0_1 = Dropout(0.1)(pool0_1)
pool2 = MaxPooling2D(pool_size=(2,2),strides=None)(drop0_1)


# FLAT

flat_0= Flatten()(pool2) #conv1_1
drop_flat = Dropout(0.5)(flat_0)


# OUTPUT
output_font = Dense(11, activation='softmax' , name='output_font')(drop_flat)
output_char = Dense(94, activation='softmax' , name='output_char')(drop_flat)
output_bold = Dense(1, activation='sigmoid' , name='output_bold')(drop_flat)
output_italics = Dense(1, activation='sigmoid' , name='output_italics')(drop_flat)
	

# Model
model= Model(inputs=visible_0, outputs= [output_font,output_char,output_bold,output_italics])


# Define the optimizer
adam = Adam()


# Define the compile of the model
model.compile(loss={"output_char":'categorical_crossentropy',"output_font":'categorical_crossentropy',"output_bold":'binary_crossentropy',"output_italics":'binary_crossentropy'}, optimizer = adam, metrics=['accuracy'])

# Print the summary of the model
model.summary()

# Train the model
model.fit(x=letter_aug, y=label, batch_size= 3000 , epochs=20, validation_split=0.2)




# Save the model

with open('../data_out/model/modelsummary.txt', 'w') as f:

    with redirect_stdout(f):

        model.summary()



model_json = model.to_json()

with open("../data_out/model/model.json", "w") as json_file:

    json_file.write(model_json)




model.save_weights("../data_out/model/model.h5")






##################################################
# Print images


lettera=letter_aug[4].reshape((1,64,64,1))

# Print images for the first convolutional layer
model_new = Model(inputs=visible_0, outputs=conv0) 
model_new.layers[0].set_weights(model.layers[0].get_weights())

#create intermediate image
convolved_single_image = model_new.predict(lettera)
convolved_single_image = convolved_single_image[0]

#plot the output of each intermediate filter
for i in range(128):
        filter_image = convolved_single_image[:,:,i]
        plt.subplot(16,8,i+1)
        plt.imshow(filter_image,cmap='gray'); plt.axis('off');
plt.savefig("../data_out/img/img_intermediate_1.png")      
plt.close()






# Print images for the first max pooling
model_new = Model(inputs=visible_0, outputs=output_font) 
model_new.layers[0].set_weights(model.layers[0].get_weights())
#create intermediate image
convolved_single_image = model_new.predict(lettera)
#plot the output of each intermediate filter
filter_image = convolved_single_image
plt.imshow(filter_image,cmap='gray'); plt.axis('off');
plt.savefig("../data_out/img/img_intermediate_2.png")      
plt.close()
        
        
        