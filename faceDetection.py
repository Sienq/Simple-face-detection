import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
camera = cv2.VideoCapture(0)


while camera.isOpened():

    ret,frame = camera.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5,minSize = (120,120))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        grayROI = gray[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(grayROI,1.03,8,minSize = (40,40))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(255,0,0),1)

    if ret:
        cv2.imshow('no ten',frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    

camera.release()
cv2.destroyAllWindows()