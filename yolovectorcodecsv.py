import cv2
import numpy as np
import csv
import math

# Load Haar Cascade for initial detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load YOLOv3 model for classification
net = cv2.dnn.readNet('face-yolov3-tiny_41000.weights', 'face-yolov3-tiny.cfg')

# Load COCO class names
with open('obj.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load CSV file with object class, coordinates, and vectors
objects = []
with open('objects.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        objects.append(row)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face ROI for YOLOv3 classification
        face_roi = frame[y:y+h, x:x+w]

        # Create a blob from the face ROI and perform forward pass with YOLOv3
        blob = cv2.dnn.blobFromImage(face_roi, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward()

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Get coordinates for drawing
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)

                    # Calculate coordinates of the bounding box
                    x = int(center_x - width / 2) + x
                    y = int(center_y - height / 2) + y

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                    # Draw class label
                    class_name = classes[class_id]
                    cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Get vector information from the CSV file
                    for obj in objects:
                        if obj['class'] == class_name:
                            vector_x = int(obj['vector_x'])
                            vector_y = int(obj['vector_y'])

                            # Calculate the end point of the vector
                            end_x = x + vector_x
                            end_y = y + vector_y

                            # Draw the vector
                            cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Object Detection with Vectors', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

