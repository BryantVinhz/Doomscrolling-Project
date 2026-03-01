import cv2
import mediapipe as mp
import pygame
import time

pygame.mixer.init()
alert_sound = pygame.mixer.Sound("warning.mp3")

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

scroll_start = None
last_alert = 0

DISTRACT_TIME = 3
ALERT_COOLDOWN = 5

calibration_start = time.time()
baseline_diff = []
DOWN_THRESHOLD = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    status = "CALIBRATING..."
    color = (0,255,255)

    if results.detections:
        for detection in results.detections:

            mp_drawing.draw_detection(frame, detection)

            keypoints = detection.location_data.relative_keypoints

            left_eye = keypoints[0]
            right_eye = keypoints[1]
            nose = keypoints[2]

            eye_y = (left_eye.y + right_eye.y) / 2
            nose_y = nose.y
            diff = nose_y - eye_y

            # ===== CALIBRATION 2 GIÂY ĐẦU =====
            if time.time() - calibration_start < 2:
                baseline_diff.append(diff)
            elif DOWN_THRESHOLD is None:
                avg = sum(baseline_diff) / len(baseline_diff)
                DOWN_THRESHOLD = avg + 0.03   # thêm margin an toàn
                print("Calibrated threshold:", DOWN_THRESHOLD)

            # ===== Sau khi calibrate =====
            if DOWN_THRESHOLD is not None:
                status = "FOCUSED"
                color = (0,255,0)

                looking_down = diff > DOWN_THRESHOLD

                if looking_down:
                    status = "DISTRACTED"
                    color = (0,0,255)

                    if scroll_start is None:
                        scroll_start = time.time()

                    elapsed = time.time() - scroll_start

                    if elapsed > DISTRACT_TIME:
                        now = time.time()
                        if now - last_alert > ALERT_COOLDOWN:
                            alert_sound.play()
                            last_alert = now
                else:
                    scroll_start = None

    cv2.putText(frame, status, (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Detection Doomscrolling", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()