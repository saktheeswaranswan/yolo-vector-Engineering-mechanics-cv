import cv2
import numpy as np
import os

# Create a folder to store binarized images
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

# Initialize dictionary to store detected objects
detected_objects = {}

# Initialize ID counter
id_counter = 0

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

        # Crop the detected object only if the bounding box is valid
        if x >= 0 and y >= 0 and width > 0 and height > 0:
            detected_object = frame[y:y + height, x:x + width]
            # Convert the cropped object to grayscale
            grayscale_object = cv2.cvtColor(detected_object, cv2.COLOR_BGR2GRAY)
            # Apply binary thresholding
            _, binarized_object = cv2.threshold(grayscale_object, 128, 255, cv2.THRESH_BINARY)
            # Generate a unique ID for the object
            object_id = f"{classes[class_id]}_{id_counter}"
            # Save the binarized object in the output folder with the unique ID
            cv2.imwrite(os.path.join(output_folder, f"{object_id}.png"), binarized_object)
            # Increment the ID counter
            id_counter += 1

    # Display the frame
    cv2.imshow('Object Detection with Binarization and NMS', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

