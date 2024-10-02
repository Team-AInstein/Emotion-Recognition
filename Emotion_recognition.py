import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import time


emotion_model = load_model('emotion_model.h5', compile=False)

emotion_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(face):
    face = cv2.resize(face, (64, 64))  
    face = face / 255.0  
    face = np.reshape(face, (1, 64, 64, 1))  
    
    predictions = emotion_model.predict(face)  
    max_index = np.argmax(predictions[0])  
    return emotion_labels[max_index], np.max(predictions[0]) 


def draw_confidence_bar(frame, x, y, w, emotion, intensity):
    
    cv2.rectangle(frame, (x, y + h + 10), (x + w, y + h + 40), (0, 0, 255), 2)
    cv2.rectangle(frame, (x, y + h + 10), (x + int(w * intensity), y + h + 40), (0, 255, 0), -1)
    cv2.putText(frame, f'{emotion} ({int(intensity * 100)}%)', (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


cap = cv2.VideoCapture(0)  

# Load the pre-trained OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()  
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    
   
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    for (x, y, w, h) in faces:
        if w < 64 or h < 64: 
            continue
        
        face = gray_frame[y:y+h, x:x+w]  
        emotion, intensity = predict_emotion(face)  
        
       
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        
        draw_confidence_bar(frame, x, y, w, emotion, intensity)
    
   
    cv2.imshow('Real-Time Emotion Recognition (Mirrored)', frame)
    
   
    time.sleep(0.03) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break


cap.release()
cv2.destroyAllWindows()
