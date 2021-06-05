import cv2
import numpy as np
import face_recognition 
import os
import re

path = 'C:/Users/HP/Desktop/face_recognition/images/'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)
for cls in myList:
  curImg = cv2.imread(f'{path}/{cls}')
  images.append(curImg)
  classNames.append(os.path.splitext(cls)[0])

#print(classNames)


def findEncodings(images):
  encodeList = []
  for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeList.append(encode)
  return encodeList
  
  
encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('encoding complete')

cap = cv2.VideoCapture(0)
#print(cap)
while True:
  success, img = cap.read()
  #imgS = cv2.resize(img,(0,0),None, 0.25,0.25)
  imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  facesCurFrame = face_recognition.face_locations(imgS)
  encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

  for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
    #print(faceDis)
    matchIndex = np.argmin(faceDis)

    if matches[matchIndex]:
      name = classNames[matchIndex].upper()
      #print(name)
      y1,x2,y2,x1 = faceLoc
      #print(y1,x2,y2,x1)
      y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
      cv2.rectangle(img,(faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]),(255,0,255),2)
      cv2.rectangle(img, (faceLoc[3],faceLoc[2]+35),  (faceLoc[1],faceLoc[2]),(255,0,255), cv2.FILLED)
      cv2.putText(img,name,(faceLoc[3]+6, faceLoc[2]+30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
      
  cv2.imshow('webcam',img)
  cv2.waitKey(1)

  
