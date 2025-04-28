# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def midpoint(ptA, ptB):
    """Calculate the midpoint between two points."""
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def process_frame(frame):
    """
    Process a single frame to detect objects and measure their dimensions in pixels.

    Args:
        frame (np.array): The current video frame.

    Returns:
        np.array: Processed frame with object dimensions and classifications overlaid.
    """
    # Convert the frame to grayscale and blur it slightly
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Perform edge detection, then perform a dilation + erosion to close gaps between edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort the contours from left-to-right
    (cnts, _) = contours.sort_contours(cnts)

    # Loop over the contours individually
    for c in cnts:
        # If the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        # Compute the rotated bounding box of the contour
        orig = frame.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv3() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # Order the points in the contour
        box = perspective.order_points(box)
        cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)

        # Loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Unpack the ordered bounding box and compute the midpoints
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # Draw the midpoints on the image
        cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # Draw lines between the midpoints
        cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # Compute the Euclidean distances between midpoints
        height = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  # Height is vertical
        width = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))    # Width is horizontal
        diameter = dist.euclidean(tl, br)  # Diagonal of the bounding box (fixed here)
        length = max(height, width)        # Length is the longer dimension

        # Classify the object based on its dimensions
        classification = "Small"
        if length > 200:
            classification = "Medium"
        if length > 400:
            classification = "Large"

        # Draw the object sizes and classification on the image
        cv2.putText(frame, f"Height: {int(height)}px", (int(tltrX - 50), int(tltrY - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Width: {int(width)}px", (int(trbrX + 10), int(trbrY + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Diameter: {int(diameter)}px", (int(tltrX - 50), int(tltrY + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Class: {classification}", (int(tltrX - 50), int(tltrY + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(4)
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

        # Process the frame and display the results
        frame = process_frame(frame)
        cv2.imshow("Real-Time Object Classification", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()