import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('C:\\Users\\USER\\Desktop\\robocar-main\\Adwaith_Hari\\Smart_Speaker\\model.h5')
cascade = cv2.CascadeClassifier(r'C:\Users\USER\Desktop\robocar-main\Adwaith_Hari\Smart_Speaker\\haarcascade_frontalface_default.xml')
tdg = ImageDataGenerator(rescale=1./255)
trgr = tdg.flow_from_directory(
    'C:\\Users\\USER\\Desktop\\robocar-main\\Adwaith_Hari\\Smart_Speaker\\images\\',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical'
)

def recognize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    for (x,y,w,h) in faces:
        face_roi = img[y:y+h, x:x+w]
        resized = cv2.resize(face_roi, (224,224))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1,224,224,3))
        result = model.predict(reshaped)
        label = trgr.class_indices
        label = dict((v,k) for k,v in label.items())
        predicted = label[np.argmax(result)]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 3)
        cv2.putText(img, str(predicted), (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,255,0), 2)
    return img

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    recognize(frame)
    
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()