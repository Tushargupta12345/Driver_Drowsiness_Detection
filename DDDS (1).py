import cv2
import dlib
import numpy as np  # Add this import statement
from scipy.spatial import distance as dist

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load the face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shapeprediction.dat")

# Define the indexes for the left and right eyes
(l_start, l_end) = (42, 48)
(r_start, r_end) = (36, 42)

# Load the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear_avg = (ear_left + ear_right) / 2.0

        # Convert eye landmarks to NumPy arrays
        left_eye_np = np.array(left_eye, dtype=np.int32)
        right_eye_np = np.array(right_eye, dtype=np.int32)

        # Draw the eyes on the frame
        cv2.polylines(frame, [left_eye_np], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_np], True, (0, 255, 0), 1)

        if ear_avg < 0.25:
            cv2.putText(frame, "Drowsy!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
