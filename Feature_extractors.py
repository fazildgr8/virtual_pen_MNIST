import cv2
import numpy as np
cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyecascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


while True:
    ret, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 70)
    ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)

    faces = faceCascade.detectMultiScale(np.copy(gray),1.1,4)
    face_img = np.copy(img)
    for (x, y, w, h) in faces:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = face_img[y:y+h, x:x+w]
        eyes = eyecascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),1)

    corner = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    corner = cv2.dilate(corner,None)
    corner_img = np.copy(img)
    # Threshold for an optimal value, it may vary depending on the image.
    corner_img[corner>0.001*corner.max()]=[0,0,255]




    cv2.imshow('Canny Edge Detected', mask)
    cv2.imshow('Corner Detection', corner_img)
    cv2.imshow('Frontal Face', face_img)
    cv2.imshow('Original', img)
    
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()