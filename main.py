import cv2
import numpy as np
from keras.models import load_model
from utils import preprocess_face
from mediapipe_facemesh import FaceMeshDetector

# Load model and labels
model = load_model("models/best_emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam and facemesh detector
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mesh_image, bbox = detector.detect_mesh(frame)

    if bbox:
        x_min, y_min, x_max, y_max = bbox
        face_roi = frame[y_min:y_max, x_min:x_max]
        processed_face = preprocess_face(face_roi)

        # Predict emotion
        prediction = model.predict(np.expand_dims(processed_face, axis=0), verbose=0)
        emotion_label = emotion_labels[np.argmax(prediction)]

        # Draw rounded bounding box
        color = (255, 255, 255)  # White box
        thickness = 2
        cv2.rectangle(mesh_image, (x_min, y_min), (x_max, y_max), color, thickness, cv2.LINE_AA)

        # Display emotion
        cv2.putText(mesh_image, emotion_label, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Emotion Detection", mesh_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
