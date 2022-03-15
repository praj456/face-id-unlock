import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('C:/Program Files/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_extracter(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  faces = face_classifier.detectMultiScale(gray,1.3,5)

  if faces is():
    return None
  else:
    for(x,y,w,h) in faces:
      croped_faces =img[y:y+h,x:x+w]
  
  return croped_faces 



cap= cv2.VideoCapture(0)

count=0

while True:
  ret,frame = cap.read()
  if face_extracter(frame) is not None:
    count+=1
    face= cv2.resize(face_extracter(frame),(250,250))

    face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

    file_name_path = 'C:/Users/prajwal/Downloads/facerecognition/images/User'+str(count)+'.jpg'
    cv2.imwrite(file_name_path,face)

    cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow('Face Cropper',face)

  else:
    print('face not found')
    pass

  if cv2.waitKey(1)==13 or count==1000:
    break

cap.release()
cv2.destroyAllWindows()
print('collected samples successfully')