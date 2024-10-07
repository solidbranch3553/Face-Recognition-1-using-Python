import cv2
import os

cap = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier(r'/home/pi/Desktop/Smart_Speaker/haarcascade_frontalface_default.xml')
count = 0
nameId = str(input("Item: ")).lower()
path = '/home/pi/Desktop/Smart_Speaker/Images/'+nameId
isExist = os.path.exists(path)

if isExist:
    print("Error!")
else:
    os.makedirs(path)

while True:
    frame = cap.read()
    faces = face_detect.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
        count = count + 1
        name = '/home/pi/Desktop/Smart_Speaker/Images/'+nameId+'/'+str(count)+'.jpg'
        print("Creating Images......"+name)
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    cv2.imshow("Window", frame)
    if count > 20:
        break
cv2.destroyAllWindows()
