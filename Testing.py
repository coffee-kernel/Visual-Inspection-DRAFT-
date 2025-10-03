import cv2
import time

# GStreamer pipeline for Acasis VC-005 (MJPG, 1920x1080@30fps, partial GPU)
DEVICE_PATH = '/dev/video0'
pipeline = (
    f"v4l2src device={DEVICE_PATH} ! "
    "image/jpeg,width=1920,height=1080,framerate=30/1 ! "
    "jpegdec ! video/x-raw ! "  # CPU-based JPEG decoding
    "nvvidconv ! video/x-raw,format=BGRx ! "  # GPU-based format conversion
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink"
)

# Initialize OpenCV with GStreamer
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print(f"Error: Could not open device {DEVICE_PATH}. Test: gst-launch-1.0 v4l2src device=/dev/video0 ! image/jpeg,width=1920,height=1080,framerate=30/1 ! jpegdec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! autovideosink")
    exit()

print("Press 'q' to QUIT.")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Display FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("SHARPEYE - VC", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()