from keras.models import model_from_json
from sys import argv
import numpy as np
import csv
import string
from keras.models import Model



# Read from user input

garbage, file_path, out_file = argv


# Load the keras model

json_file = open('../data_out/model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../data_out/model/model.h5")




# Load 
X_data = np.load(file_path)


for immagine in range(len(X_data)):
    aux = X_data[immagine]
    minimo=np.min(aux)
    massimo=np.max(aux)
    if massimo != 0.0:
        X_data[immagine] = (aux - minimo)/massimo

# X_data= np.load('../data/data_clean.npy')


#loaded_model.compile(loss=['categorical_crossentropy','categorical_crossentropy','binary_crossentropy','binary_crossentropy'], optimizer = adam, metrics=['accuracy'])

print(loaded_model.summary())

#score = loaded_model.evaluate(X, Y, verbose=0)




# save the predictions as a csv file into ../data_out/test/OUTPUT_FILE_NAME.csv, one per each row ( no header):
ynew = loaded_model.predict(X_data, verbose = 1)

path = '../data_out/test/'+out_file + '.csv'

alphabet = list(string.printable[:-6])
list_font = np.load('../data/font_name.npy')


with open(path, 'w') as myfile:
    wr = csv.writer(myfile)
    for r in range(len(ynew[0])):
        fontt = list_font[list(ynew[0][r]).index(max(list(ynew[0][r])))]
        charr = alphabet[list(ynew[1][r]).index(max(list(ynew[1][r])))]
        
        if ynew[2][r]>0.5:
            italicss = 1.0
        else: 
            italicss = 0.0
            
        if ynew[3][r]>0.5:
            boldd = 1.0
        else: 
            boldd = 0.0
        mylist = [charr, fontt,boldd, italicss]

        wr.writerow(mylist)
        
        