import cv2
import numpy
import face_recognition

imaron= face_recognition.load_image_file('ImageBasic/bunny.jpeg')
imaron=cv2.cvtColor(imaron,cv2.COLOR_BGR2RGB)

imates= face_recognition.load_image_file('ImageBasic/phani.jpeg')
imates=cv2.cvtColor(imates,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imaron)[0]
encoderon=face_recognition.face_encodings(imaron)[0]
cv2.rectangle(imaron,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


faceLoctest=face_recognition.face_locations(imates)[0]
encodetest=face_recognition.face_encodings(imates)[0]
cv2.rectangle(imates,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encoderon],encodetest)
faceDis=face_recognition.face_distance([encoderon],encodetest)
print(results,faceDis)
cv2.putText(imates,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Match1',imaron)
cv2.imshow('Match2 Test',imates)
cv2.waitKey(0)
