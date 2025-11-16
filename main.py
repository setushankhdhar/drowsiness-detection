import cv2
import mediapipe as mp
import numpy as np
import threading
from playsound import playsound  # ✅ replaced simpleaudio
from utils.eye_utils import eye_aspect_ratio

# Initialize Mediapipe FaceMesh safely
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Thresholds
EAR_THRESHOLD = 0.25
ALERT_FRAMES = 10  # number of continuous frames for drowsiness
closed_frames = 0
alarm_playing = False

# ✅ Function to play alarm in background
def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        try:
            playsound("alarm/alarm.wav")
        except Exception as e:
            print(f"⚠️ Error playing alarm: {e}")
        alarm_playing = False

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Unable to access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        mesh_points = np.array([[p.x, p.y] for p in results.multi_face_landmarks[0].landmark])
        h, w, _ = frame.shape
        mesh_points = np.multiply(mesh_points, [w, h]).astype(int)

        left_EAR = eye_aspect_ratio(mesh_points, LEFT_EYE)
        right_EAR = eye_aspect_ratio(mesh_points, RIGHT_EYE)
        EAR = (left_EAR + right_EAR) / 2.0

        if EAR < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0

        # ✅ Trigger as soon as the DROWSY condition appears
        if closed_frames >= ALERT_FRAMES:
            cv2.putText(frame, "DROWSY!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3)
            
            # Start alarm instantly (thread prevents blocking)
            threading.Thread(target=play_alarm, daemon=True).start()

        else:
            cv2.putText(frame, "ACTIVE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit when 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
