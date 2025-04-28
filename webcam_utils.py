import cv2
import numpy as np

def calculate_measurements(contour):
    """
    Calculate the length, width, and diameter of an object based on its bounding box.
    """
    # Get the bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Length and width are the dimensions of the bounding box
    length = h  # Height of the bounding box
    width = w   # Width of the bounding box
    
    # Diameter is the diagonal of the bounding box
    diameter = int(np.sqrt(w**2 + h**2))
    
    return length, width, diameter, (x, y, w, h)

def draw_measurements(frame, x, y, w, h, length, width, diameter):
    """
    Draw measurement lines and labels on the frame.
    """
    # Draw the bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw measurement lines
    cv2.line(frame, (x, y), (x, y + h), (255, 0, 0), 2)  # Length
    cv2.line(frame, (x, y + h), (x + w, y + h), (0, 0, 255), 2)  # Width
    cv2.line(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Diameter
    
    # Display measurement labels
    cv2.putText(frame, f"Length: {length}px", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Width: {width}px", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Diameter: {diameter}px", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    # Open webcam
    cap = cv2.VideoCapture(4)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit the program.")
    
    while True:
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Convert frame to grayscale and apply threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter out small contours (noise)
            if cv2.contourArea(contour) > 500:
                # Calculate measurements
                length, width, diameter, (x, y, w, h) = calculate_measurements(contour)
                
                # Draw measurements on the frame
                draw_measurements(frame, x, y, w, h, length, width, diameter)
        
        # Display the frame
        cv2.imshow("Real-Time Measurements", frame)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()