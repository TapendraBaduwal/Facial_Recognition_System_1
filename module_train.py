##Train Module#
import cv2                     #opencv library import as cv2
import numpy as np             # numpy library for numerical calculation    
from os import listdir         #os listdir() method in python is used to get the list of all files and directories in the specified directory. 
from os.path import isfile,join #The os.path.isfile() method returns True if the specified path is an existing regular file. Otherwise, it returns False.



data_path="/home/madan/Desktop/images/"  #path of image data 

onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))] #apply for loop,since files of images are strores in list form so use [], pass  data_path in listdir,apply if condition  if file is found then join data_path with f.
Training_Data,Labels=[],[]
for i, files in enumerate(onlyfiles):    #enumerate provide iteration upto files number
    image_path=data_path + onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype=np.uint8))#append images in the form of array
    Labels.append(i)                                       # append i values in Labels
Labels=np.asarray(Labels,dtype=np.int32)                   #call as asarray and pass labels and data type np.int32 bit
model=cv2.face.LBPHFaceRecognizer_create()                 #Built a module and call face.LBPHFaceRecognizer 
model.train(np.asarray(Training_Data),np.asarray(Labels))  #now train module by passing Training_data through np.asarray ad pass Labels through np.asarray.
print("Tapendra your model training complet")