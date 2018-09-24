from PIL import ImageDraw , ImageFont
from PIL import Image
import glob
import numpy as np
import re
import string


# Define all the possible characters that can be used in the images
alphabet = list(string.printable[:-6])

# Define the path in which the fonts file are stored
folder_path= '../data/fonts'

# Extract, from the selected path, all the file with .ttf extension
path2= folder_path +'/*.ttf'
font_path= glob.glob(path2)

# Extract, from the selected path, all the file with .otf extension
path3= folder_path +'/*.otf'
font_path2= glob.glob(path3)

# Combine the two lists created before
font_path += font_path2

# Initialize empty lists
out=[] 
labels=[]
font_mat= []
italics=[]
bold=[]
char_=[]

# Start to iterate over the different fonts
_id=0
for key in font_path:

	# Define, from the selected path, if the used font is bold and/or italics
    match_B_I= bool(re.search('_B_I.ttf', key))
    match_B_I_bis= bool(re.search('_B_I.otf', key))
    match_B= bool(re.search('_B', key))
    match_I= bool(re.search('_I',key))
    
    fnt = ImageFont.truetype(key, 40)
    
    # Iterate over the number the characters
    for let in alphabet:

        
        # Draw the character on the image
        txt = Image.new('L', (64,64), (0))
        d = ImageDraw.Draw(txt) 
        w, h = d.textsize(let, font = fnt)
        d.text((((64-w)/2),(64-h/2)-32), let, font=fnt, fill=255)
        if np.max(txt)== 255:
            out.append(np.array(txt))
            
            # Select the index related to the selected char, in order to create the labels
            char_aux=[0 for i in range(len(alphabet))]
            char_aux[alphabet.index(let)]= 1
            char_.append(np.array(char_aux))
            
            # Detect if the image is bold AND italics and create the related labels
            if match_B_I ==True or match_B_I_bis==True:
                font1= re.search('fonts\/(.+?)_', key)
                labels.append( {'output_font':font1.group(1),'output_char' :let,'output_bold' : 1,'output_italics': 1} )
                font_mat.append(font1.group(1))
                bold.append(1)
                italics.append(1)
                
            # Detect if the image is ONLY italics and create the related labels
            elif match_I==True:
                font2= re.search('fonts\/(.+?)_', key)
                labels.append( {'output_font':font2.group(1),'output_char' :let,'output_bold' : 0,'output_italics': 1} )
                font_mat.append(font2.group(1))
                bold.append(0)
                italics.append(1)
            
            # Detect if the image is only BOLD and create the related labels
            elif match_B==True:
                font3= re.search('fonts\/(.+?)_', key)
                labels.append( {'output_font':font3.group(1),'output_char' :let,'output_bold' : 1,'output_italics': 0} )            
                font_mat.append(font3.group(1))
                bold.append(1)
                italics.append(0)
             
            # Detect if the image is neither bold nor italics    and create the related labels
            else:
                if key[-3:]=='ttf':
                    font4= re.search('fonts\/(.+?).ttf', key)
                else:
                    font4= re.search('fonts\/(.+?).otf', key)
                labels.append( {'output_font':font4.group(1),'output_char' :let,'output_bold' : 0,'output_italics': 0} )            
                font_mat.append(font4.group(1))            
                bold.append(0)
                italics.append(0)        
            
            _id+=1
     


font_list = list(set(font_mat))
auxx=[]
for i in labels:
    aux= [0 for i in range(len(font_list))]
    aux[font_list.index(i['output_font'])]=1
    auxx.append(np.array(aux))
    
italics= np.array(italics)
char_= np.array(char_)
bold= np.array(bold)    
auxx= np.array(auxx)    
labels= np.array(labels)
letters= np.array(out, dtype='float64' )                    
letters2= letters.reshape(_id,64,64,1)






# Create a final dictionary of labels
labels3={'output_font':auxx , 'output_bold': bold ,'output_char': char_ ,'output_italics': italics }

# Save data and labels in two different .npy files
np.save('../data/data_clean',letters2)
np.save('../data/label',labels3)


# Save the font list

font_list = np.array(font_list)
np.save('../data/font_name',font_list)
