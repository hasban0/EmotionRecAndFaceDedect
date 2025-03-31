# pip install -r requirements.txt

import cv2
import numpy as np
from keras.models import model_from_json

# Emotion labels and their numerical counterparts
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# Read the emotion model's JSON file and load the model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)


emotion_model.load_weights("model/emotion_model.h5")
print("Model loaded from disk..")

# Start the camera source
# To use a USB camera, modify the code to cap = cv2.VideoCapture(0).
cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    # Load face detector classifier
    face_detector = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces on the grayscale frame
    num_faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

# Draw rectangles around the detected faces
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

    # Make emotion prediction
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20),  # Write the emotion on the rectangle
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
# Display results on screen
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
