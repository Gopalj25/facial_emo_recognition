# mediapipe_facemesh.py
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

class FaceMeshDetector:
    def __init__(self, static_mode=False, max_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.draw_spec = mp_drawing.DrawingSpec(
            thickness=1, circle_radius=0, color=(144, 238, 144)  # Light green
        )

    def detect_mesh(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        mesh_image = image.copy()
        bbox = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw thin mesh lines with small dots
                mp_drawing.draw_landmarks(
                    image=mesh_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    #connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.draw_spec,
                    connection_drawing_spec=self.draw_spec
                )
                # Bounding box calculation
                img_h, img_w = image.shape[:2]
                x_coords = [int(lm.x * img_w) for lm in face_landmarks.landmark]
                y_coords = [int(lm.y * img_h) for lm in face_landmarks.landmark]
                x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, img_w)
                y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, img_h)
                bbox = (x_min, y_min, x_max, y_max)

        return mesh_image, bbox
