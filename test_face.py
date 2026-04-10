import cv2
import mediapipe as mp

# Initialize drawing tools and Face Mesh of MediaPipe
mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Configure how to draw landmarks (green color, thin lines)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize Face Mesh AI model
with mp_face_mesh.FaceMesh(
    max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as face_mesh:
    print("Face Mesh is running. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from camera!")
            break

        # Convert BGR → RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run AI face detection
        results = face_mesh.process(rgb_frame)

        # Process results
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=draw_spec,
                    connection_drawing_spec=draw_spec,
                )
            status_text = "Face Detected"
            status_color = (0, 255, 0)
        else:
            status_text = "No Face"
            status_color = (0, 0, 255)

        # Display status on image
        cv2.putText(
            frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2
        )

        # Display image
        cv2.imshow("Face Mesh - Doomscrolling Project", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
