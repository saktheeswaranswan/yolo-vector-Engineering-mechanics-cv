import cv2
import numpy as np
import os

# Create a folder to store grayscale images
output_folder = 'output_objects'
os.makedirs(output_folder, exist_ok=True)

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

# Define grid properties
grid_color = (255, 255, 255)  # White color for the grid
grid_step = 20  # Adjust this value to control the grid density

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

    # Apply Non-Maximum Suppression to remove duplicate bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    for i in indices:
        i = i
        class_id = class_ids[i]
        box = boxes[i]
        x, y, width, height = box

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Use Hershey Simplex font for labeling
        label = f"{classes[class_id]}: {confidences[i]:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Create a grid within the bounding box
        for dx in range(0, width, grid_step):
            cv2.line(frame, (x + dx, y), (x + dx, y + height), grid_color, 1)
        for dy in range(0, height, grid_step):
            cv2.line(frame, (x, y + dy), (x + width, y + dy), grid_color, 1)

        # Crop the detected object and save it as grayscale
        detected_object = frame[y:y + height, x:x + width]
        grayscale_object = cv2.cvtColor(detected_object, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_folder, f"{classes[class_id]}_{i}.png"), grayscale_object)

    # Display the frame
    cv2.imshow('Object Detection with Grid and NMS', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

