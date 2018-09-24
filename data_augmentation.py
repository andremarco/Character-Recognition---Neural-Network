from skimage.transform import *
from skimage.util import pad, random_noise
import numpy as np
import tensorflow as tf
from math import ceil, floor, pi
from PIL import Image
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# Load the base dataset and the related labels
letter_base= np.load('../data/data_clean.npy')
label_normal = np.load('../data/label.npy').item()

####################################################################################################################
####################################################################################################################

# Define functions that will be used to augment the data

# RESCALE

def rescale_MM(image):
	# Zooming images
	
	scale=random.uniform(1.6,1.95) 
    resi=rescale(image, scale, preserve_range=True)
	## the function rescale rescales also the margins so we need to fill with zeros to obtain the starting dimensions
	ret=int((resi.shape[0]-image.shape[0])/2)
	rett=int(image.shape[0]+ ret)
	arr_PP = resi[ret:rett,ret:rett,:]
    arr_PP = arr_PP.reshape(64,64,1)
    return(arr_PP)



# ROTATE

def rotate_MM(image):
    angolo = random.randint(-33,33)
    rot=rotate(image, angolo, mode="constant", cval=0)
    return(rot)


# RANDOM NOISE 
def random_noise_MM(image):
    row,col,ch= image.shape
    mean = 0.004
    var = 0.00000000002
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy = abs(noisy)
    return noisy


# SALT and PEPPER

def pepper_MM(image):
 
    altezza = [random.randint(0,63) for i in range(110)]
    base = [random.randint(0,63) for i in range(110)]
    for i in range(110):
        image[altezza[i]][base[i]] = 0
            
    image = image/255
    return(image)


# TRASLATION

def traslation_MM(image):
    tx = random.randint(-4,4)
    ty = random.randint(-4,4)
    transform = AffineTransform(translation=(tx,ty))
    shifted = warp(image, transform, mode='constant', preserve_range=True, cval = 0)
    return(shifted)



# Create a function that allow us to augment all the basic images using all the functions defined above

def augment_images(X_imgs, labe_norm, n_copies = 20):
    '''This function create a random numbers of augmented images'''

    # Define a list with some usable functions
    functions = [traslation_MM, rotate_MM, rescale_MM] 
	# saving into already the number of basic images
    already = len(X_imgs)
    copia = np.zeros((already,64,64,1))
    
    
    num = 0

    indicone = 0
    num_tot = n_copies * already
	# saving into fgg  matrices of zeros. The amount of matrices is the number of final images.
    fgg = np.zeros((num_tot,64,64,1))
    
	
    old_char = labe_norm["output_char"]
    old_bold = labe_norm["output_bold"]
    old_italics = labe_norm["output_italics"]
    old_font = labe_norm["output_font"]
    
    new_char = np.ones((num_tot,94))
    new_bold = np.ones((num_tot))
    new_italics = np.ones((num_tot))
    new_font = np.ones((num_tot,11))
    
    

    # Iterate over all images
    for img in range(len(X_imgs)):
		# create labels for augmented images
        aux_bold = labe_norm["output_bold"][img]
        aux_italics = labe_norm["output_italics"][img]
        aux_char = labe_norm["output_char"][img]
        aux_font = labe_norm["output_font"][img]
        
		# switch image's colors in order to obtain white background.
        aux = abs(((X_imgs[img]/255) - 1)*(-1))
        copia[indicone] = aux
        indicone +=1

		# iterate over the number of copies for each image
        for copi in range(n_copies):
		    # Matching the image with its label
            new_char[num] = aux_char
            new_font[num] = aux_font
            new_bold[num] = aux_bold
            new_italics[num] = aux_italics



            # Define n_fun that is the number of functions to apply to each image
            n_fun = random.randint(1,2)
            
            # Create a copy
            img_copy= X_imgs[img]
			
            # randomly select functions to apply
            for sam_fun in random.sample(functions, n_fun):
      
                img_copy= sam_fun(img_copy)
            
            # probabilty to apply salt and pepper
            prob_salt = random.uniform(0,1)

			if prob_salt > 0.35:
                img_copy = pepper_MM(img_copy)
            else:
                img_copy = img_copy/255

            # probabilty to apply random_noise				
            prob_random_noise = random.uniform(0,1)
            if prob_random_noise > 0.9:
                img_copy = random_noise_MM(img_copy)
            
            # switch image's colors in order to obtain white background.
            img_copy = abs((img_copy -1)*(-1))
            fgg[num] = img_copy

            
            num+=1
            print(num)
			
	# create the final dataset
	
    fgg = np.concatenate((copia, fgg))
	
    final_char = np.concatenate((old_char,new_char))
    final_bold = np.concatenate((old_bold,new_bold))
    final_italics = np.concatenate((old_italics,new_italics))
    final_font = np.concatenate((old_font,new_font))    
    final_lab={'output_font':final_font , 'output_bold': final_bold ,'output_char': final_char ,'output_italics': final_italics }

    
    return([fgg, final_lab])



data_augmentate, label_augmentate = augment_images(letter_base, label_normal, n_copies=100)

# Normalizing images
for immagine in range(len(data_augmentate)):
    aux = data_augmentate[immagine]
    minimo=np.min(aux)
    massimo=np.max(aux)
    if massimo != 0.0:
        data_augmentate[immagine] = (aux - minimo)/massimo
		

		
		
# Save the augmented images

np.save('../data/data_augmented',data_augmentate)

np.save('../data/label_augmented', label_augmentate)






