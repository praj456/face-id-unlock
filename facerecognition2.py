import cv2
import numpy as np
import mediapipe as mp
import HandTrackingModule as htm
from os import listdir
from os.path import isfile, join


data_path = 'C:/Users/prajwal/Downloads/facerecognition/images/'
only_files=[f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_data,labels=[], []

for i, files in enumerate(only_files):
  image_path = data_path+only_files[i]
  images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
  Training_data.append(np.asarray(images,dtype=np.uint8))
  labels.append(i)
labels = np.asarray(labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_data),np.asarray(labels))
print('Model training completed successfully')


face_classifier = cv2.CascadeClassifier('C:/Program Files/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_detector(img,size=0.5):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  faces = face_classifier.detectMultiScale(gray,1.3,5)

  if faces is():
    return img,[]

  for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    region_of_interest = img[y:y+h,x:x+w]
    region_of_interest = cv2.resize(region_of_interest,(350,350))

  return img,region_of_interest



cap = cv2.VideoCapture(0)
while True:
  ret , frame = cap.read()


  image, face = face_detector(frame)
  print()

  try:

    face= cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    result = model.predict(face)

    if result[1] < 500:
      confidence =int(100*(1-(result[1])/300))
      display_string = str(confidence)+'% Confidence it is user'
    cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,120,255),2)

    if confidence >= 90:
      cv2.putText(image,'Unlocked',(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
      cv2.imshow('face Cropper',image)
    else:
      cv2.putText(image,'Locked',(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
      cv2.imshow('face Cropper',image)
  except:

    cv2.putText(image,'face not found',(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

    cv2.imshow('face Cropper',image)
    pass
  if cv2.waitKey(1)==13:
    break

cap.release()
cv2.destroyAllWindows()
