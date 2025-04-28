import cv2
import torch
from PIL import Image
import numpy as np

# Load the YOLOv5 model
# Replace 'best.pt' with the path to your trained model weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

def detect_objects(frame):
    """
    Detect specific objects in the frame using the YOLOv5 model.

    Args:
        frame (np.array): The current video frame.

    Returns:
        np.array: Processed frame with detections overlaid.
    """
    # Convert the frame to a PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform object detection
    results = model(img)

    # Extract detection results
    detections = results.xyxy[0].numpy()  # [xmin, ymin, xmax, ymax, confidence, class]

    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_id = detection
        label = results.names[int(class_id)]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        # Capture a single frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Detect objects in the frame
        frame = detect_objects(frame)

        # Display the results
        cv2.imshow("Real-Time Object Detection", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()