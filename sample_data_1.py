import cv2
#CascadeClassifier function  call which help to identify the image is face and haarcascade.xml file download from github containing face features.   
face_classifier=cv2.CascadeClassifier("haarcascade.xml")
#define function to convert image into gray,give scalefactor like 1.3,minNeighbour factor 5.
def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    for(x,y,w,h) in faces: 
        cropped_face=img[y:y+h,x:x+w] #cropped face using x coordinate,y coordinate,height and width of image
    return cropped_face

cap=cv2.VideoCapture(0) #open camera to take photo 0 for webcam live video
count=0
while True:  
    ret,frame=cap.read()  #camera read then acess through frame and value hold by ret variable
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file_name_path="images/Tapendra"+str(count)+".jpg"  #path or folder name(images) to store image,name of image,in string form,with .jpg form.
        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)#count,font ,color B=0,G(green)=255 given,R=0, thickness=2
        cv2.imshow("face cropper",face)
    else:
        print("face not found")
        pass
    if cv2.waitKey(1)==13 or count==100: #100 image or entrer press to stop #13 is askey code.
        break                            #When I use cv2.waitKey(1), I get a continuous live video feed from my laptops webcam.
cap.release()# This releases the webcam, then closes all of the imshow() windows.
cv2.destroyAllWindows()
print("collecting sample image of Tapendra Baduwal completed!!!!!")