import cv2
import numpy as np
import math

# Set your desired screen resolution here
width = 1280  # Change to your preferred width
height = 720  # Change to your preferred height

# Initialize webcam with the specified resolution
cap = cv2.VideoCapture("xcvcvxcv.mp4")
cap.set(3, width)  # Set width
cap.set(4, height)  # Set height

# Load YOLOv3 Tiny model for classification
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

# Load COCO class names (80 classes)
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

while True:
    ret, frame = cap.read()

    # Create a blob from the frame and perform forward pass with YOLOv3 Tiny
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

            if confidence > 0.5:  # Consider all detected classes with confidence > 0.5
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    # Apply Non-Maximum Suppression to suppress overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        i = i
        class_id = class_ids[i]
        box = boxes[i]
        x, y, width, height = box

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Calculate the origin (upper-left corner) of the bounding box
        origin_x = x
        origin_y = y

        # Draw class label with Hershey Simplex font
        class_name = classes[class_id]
        label = f"{class_name}: {confidences[i]:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Create a grid inside the bounding box with a step of 1 along X and Y axes
        grid_spacing = 1
        for grid_x in range(origin_x, origin_x + width, grid_spacing):
            cv2.line(frame, (grid_x, origin_y), (grid_x, origin_y + height), (255, 0, 0), 1)
            for grid_y in range(origin_y, origin_y + height, grid_spacing):
                cv2.line(frame, (origin_x, grid_y), (origin_x + width, grid_y), (255, 0, 0), 1)

        # Define vector equations for each class using the origin as the zero axis
        vector_equations = {
            'person': lambda w, h, x, y: [(x, y, x + 20, y + 20)],
            'car': lambda w, h, x, y: [(x, y, x + 30, y - 30)],
            'cat': lambda w, h, x, y: [(x, y, x - 20, y + 20)],
            'dog': lambda w, h, x, y: [(x, y, x - 30, y - 30)],
            'truck': lambda w, h, x, y: [(x, y, x + 40, y + 40)]
        }

        # Scale and plot vectors within the bounding box based on the class
        if class_name in vector_equations:
            vectors = vector_equations[class_name](width, height, origin_x, origin_y)
            for start_x, start_y, end_x, end_y in vectors:
                start_point = (int(start_x), int(start_y))
                end_point = (int(end_x), int(end_y))
                cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Object Detection with Grid and Vectors', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

