import cv2
import os

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Open the default camera
cap = cv2.VideoCapture(0)

# Get frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# First attempt to save to Desktop/Main folder
output_path = '/Users/manibharthi/Desktop/Main/output.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec

# Try to create the VideoWriter
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# If failed, fallback to Downloads folder
if not out.isOpened():
    print(f"⚠️ Failed to open VideoWriter at {output_path}. Trying fallback path...")
    output_path = '/Users/manibharthi/Downloads/output.avi'
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# If still not working, exit
if not out.isOpened():
    print(f"❌ Failed to open VideoWriter at {output_path}. Check codec or permissions.")
    cap.release()
    exit()

print(f"✅ Recording video to: {output_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from camera.")
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the frame to the output video file
    out.write(frame)

    # Show live feed
    cv2.imshow('Face Detection - Press q to exit', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Video recording complete.")