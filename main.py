import cv2
import mediapipe as mp
import numpy as np
import threading
from playsound import playsound  
from utils.eye_utils import eye_aspect_ratio
from utils.mouth_utils import mouth_aspect_ratio

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [78, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 317, 14, 87, 178, 88, 95]

EAR_THRESHOLD = 0.25
ALERT_FRAMES = 10
closed_frames = 0
alarm_playing = False

# Yawn detection
yawn_frames = 0
YAWN_FRAMES = 12
MAR_THRESHOLD = 0.82


# ------------------ FIXED ALARM FUNCTION ------------------
def play_alarm():
    global alarm_playing
    if alarm_playing:
        return    # already playing

    alarm_playing = True
    try:
        playsound("alarm/alarm.wav", block=True)   # ensures sound plays fully
    except Exception as e:
        print(f"Error playing alarm: {e}")
    finally:
        alarm_playing = False
# -----------------------------------------------------------


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to access the camera.")
    exit()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:

        mesh_points = np.array(
            [[p.x, p.y] for p in results.multi_face_landmarks[0].landmark]
        )
        h, w, _ = frame.shape
        mesh_points = (mesh_points * [w, h]).astype(int)

        left_EAR = eye_aspect_ratio(mesh_points, LEFT_EYE)
        right_EAR = eye_aspect_ratio(mesh_points, RIGHT_EYE)
        EAR = (left_EAR + right_EAR) / 2.0

        MAR = mouth_aspect_ratio(mesh_points, MOUTH)

        # Yawn detection
        if MAR > MAR_THRESHOLD:
            yawn_frames += 1
        else:
            yawn_frames = 0

        if yawn_frames >= YAWN_FRAMES:
            cv2.putText(frame, "YAWNING!", (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        # Eye-based drowsiness detection
        if EAR < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0

        if closed_frames >= ALERT_FRAMES:
            cv2.putText(frame, "DROWSY!", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # ---> Alarm plays exactly once
            if not alarm_playing:
                threading.Thread(target=play_alarm, daemon=True).start()

        else:
            cv2.putText(frame, "ACTIVE", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
