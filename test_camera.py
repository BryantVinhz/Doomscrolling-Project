import cv2

# Step 1: Open webcam
cap = cv2.VideoCapture(0)

# Step 2: Check camera is open
if not cap.isOpened():
    print("Cannot open camera!")
    exit()

print("Camera is open. Press 'D' to exit.")

# Step 3: Loop to read and display video
while True:
    # Read 1 frame from camera
    ret, frame = cap.read()

    # Check if frame is read successfully
    if not ret:
        print("Cannot read frame from camera!")
        break

    # Display frame in window
    cv2.imshow("Test Camera - Doomscrolling Project", frame)

    # Press 'D' to exit
    if cv2.waitKey(1) & 0xFF == ord("D"):
        break

# Step 4: Release resources
cap.release()
cv2.destroyAllWindows()
