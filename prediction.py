#module train
import cv2
import numpy as np
from os import listdir, truncate
from os.path import isfile,join
data_path="/home/madan/Desktop/images/"
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]
Training_Data,Labels=[],[]
for i, files in enumerate(onlyfiles):
    image_path=data_path + onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)
Labels=np.asarray(Labels,dtype=np.int32)
model=cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data),np.asarray(Labels))
print("Tapendra your model training complet")


#Prediction of data after module train
face_classifier=cv2.CascadeClassifier("haarcascade.xml")

def face_detector(img,size=0.5):    #img images come from camera frame
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():            #if face is not there then return empty image frame
        return img,[]
    for(x,y,w,h) in faces:     #for loop for roi(reason of interest) image information
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2) # received faces draw under rectangle
        roi=img[y:y+h,x:x+w]                             #roi(reason of interest)
        roi=cv2.resize(roi,(200,200))
    return img,roi
cap=cv2.VideoCapture(0)     #on camera ,0 for webcam live video
while True:
    ret,frame=cap.read()
    image,face=face_detector(frame)

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)
        if result[1]<500:                               #500 and 300 are just a value,(result[1]/300) is jst deviation from confidence
            confidence=int(100*(1-(result[1])/300))     # to calculate % of confidence face match, division give float value ,convert float to int,
            display_string=str(confidence)+"% Hello Tapendra Baduwal"
        cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        if confidence > 80:                             #if confidence is >80 then face match else face not match
            cv2.putText(image,"Face match",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow("face cropper",image)
        else:
            cv2.putText(image,"Face not match",(250,450),cv2.FONT_HERSHEY_COMPLEX,1(0,0,255),2)
            cv2.imshow("face cropper",image)
    except:
        cv2.putText(image,"Face not found",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        cv2.imshow("face cropper",image)
        pass
    if cv2.waitKey(1) & 0XFF==ord("q"): #Enter q to  to close imshow() windows
        break      #When I use cv2.waitKey(1), I get a continuous live video feed from my laptops webcam.

cap.release()
cv2.destroyAllWindows()





