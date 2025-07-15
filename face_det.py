import cv2
import insightface
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load face detector
face_detector = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# print(face_detector.models)
face_detector.prepare(ctx_id=0)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV format) to RGB (InsightFace expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_detector.get(rgb_frame)

    # Blur each face
    for face in faces:
        x1, y1, x2, y2 = [int(k) for k in face.bbox]

        # Ensure valid coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        # Extract ROI and apply Gaussian blur
        face_roi = frame[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        frame[y1:y2, x1:x2] = blurred_face

    # Show the result
    cv2.imshow("Face Blurred Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
