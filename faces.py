# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:51:23 2018

@author: Shreyas
"""

import urllib
import cv2
import numpy as np
import urllib.request
import pickle
import faces_train as tr
import os
import time



def save_im(n,name):
    img_item = image_dir+'/'+name+'/image{}.jpg'.format(n)
    print(img_item)
    cv2.imwrite(img_item, roi_color)
    #time.sleep(1)
    
def center(points):
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0
    return np.array([np.float32(x), np.float32(y)], np.float32)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

url = 'http://192.168.10.150:8080/shot.jpg?rnd=453229'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
users_online = []
user = 0
name = "User"
while(True):
    with urllib.request.urlopen(url) as u:
        cap = u.read()
    # Numpy to convert into a array
    imgNp = np.array(bytearray(cap),dtype=np.uint8)
    
    # Finally decode the array to OpenCV usable format ;) 
    frame = cv2.imdecode(imgNp,-1)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        	#print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]        
        # recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_, conf = recognizer.predict(roi_gray)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        stroke = 2
        if conf>140 : #and conf <= 85:
            
            name = labels[id_]

            if name in users_online:
                print(name)
                break
            else :
                track_window = (x, y, w, h)
                if users_online:
                    del users_online[0]
                users_online.append(name)   
        #TOdo fix lag while training faces.
        elif conf<100:
           print("Training faces now")
           user = user + 1
           millis = int(round(time.time() * 1000))
           name = str(millis)+'user{}'.format(user)
           #Add to csv # todo
           cv2.putText(frame, "Training nowq", (x,y), font, 1, color, stroke, cv2.LINE_AA)
           db = open("users.csv","w+")
           db.write(","+name)
           db.close
           BASE_DIR = os.path.dirname(os.path.abspath(__file__))
           image_dir = os.path.join(BASE_DIR, "images") 
           os.makedirs(image_dir+'/'+ name)
           for train_number in range(0,10):
               save_im(train_number,name)
           tr.train()
           print("Training is done")
        cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        color = (255, 0, 0) #BGR 0-255 
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        #save_im(i)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()