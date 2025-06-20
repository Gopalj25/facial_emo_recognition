import cv2
import numpy as np

def preprocess_face(face_image):
    """
    Preprocess the face image for CNN input:
    - Convert to grayscale
    - Resize to (48, 48)
    - Normalize pixel values to [0, 1]
    - Reshape to (48, 48, 1) if needed
    """

    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (48, 48, 1))  # For CNN expecting 1 channel

    return reshaped
