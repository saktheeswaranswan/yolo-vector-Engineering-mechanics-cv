import cv2
import numpy as np
import csv
import math
import random
import time

# Set your desired screen resolution here
width = 1280  # Change to your preferred width
height = 720  # Change to your preferred height

# Initialize webcam with the specified resolution
cap = cv2.VideoCapture(0)
cap.set(3, width)  # Set width
cap.set(4, height)  # Set height

# Load YOLOv3 model for classification
net = cv2.dnn.readNet('face-yolov3-tiny_41000.weights', 'face-yolov3-tiny.cfg')

# Load COCO class names
with open('obj.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load CSV file with object class, coordinates, angles, and vectors
objects = []
with open('objects.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        objects.append(row)

# Define the initial shrink factor and interval
shrink_factor = 0.01  # Set a very small shrink factor
shrink_interval = 10  # Shrink the vector every 10 seconds

last_shrink_time = time.time()

while True:
    ret, frame = cap.read()

    # Create a blob from the frame and perform forward pass with YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'face':
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    # Apply Simplex NMS to suppress overlapping face detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        i = i  # Extract the index from the list
        box = boxes[i]
        x, y, width, height = box

        # Get vector information from the CSV file
        for obj in objects:
            if obj['class'] == 'face':
                if 'angle' in obj and 'start_x' in obj and 'start_y' in obj and 'end_x' in obj and 'end_y' in obj:
                    angle = float(obj['angle'])
                    start_x = int(obj['start_x'])
                    start_y = int(obj['start_y'])
                    end_x = int(obj['end_x'])
                    end_y = int(obj['end_y'])

                    # Calculate the time-based scaling factor
                    scale_factor = 1.0 + (time.time() % shrink_interval) / shrink_interval * shrink_factor

                    # Scale the vector
                    scaled_end_x = int(end_x * scale_factor)
                    scaled_end_y = int(end_y * scale_factor)

                    # Rotate the vector
                    angle_rad = math.radians(angle)
                    rotated_end_x = int(start_x + math.cos(angle_rad) * scaled_end_x - math.sin(angle_rad) * scaled_end_y)
                    rotated_end_y = int(start_y + math.sin(angle_rad) * scaled_end_x + math.cos(angle_rad) * scaled_end_y)

                    # Adjust coordinates within the bounding box
                    x1 = max(x, min(x + rotated_end_x, x + width - 1))
                    y1 = max(y, min(y + rotated_end_y, y + height - 1))

                    # Draw the rotated and scaled vector within the bounding box
                    cv2.arrowedLine(frame, (x1, y1), (x1 + rotated_end_x, y1 + rotated_end_y), (0, 0, 255), 2)

                    # Update the CSV file with the new scaled vector values
                    obj['end_x'] = str(rotated_end_x)
                    obj['end_y'] = str(rotated_end_y)
                    with open('objects.csv', 'w', newline='') as csvfile:
                        fieldnames = ['class', 'angle', 'start_x', 'start_y', 'end_x', 'end_y']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for updated_obj in objects:
                            writer.writerow(updated_obj)

    # Display the frame
    cv2.imshow('Object Detection with Rotated and Scaled Vectors', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

