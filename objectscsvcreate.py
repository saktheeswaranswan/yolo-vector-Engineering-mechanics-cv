import csv
import random

# COCO dataset class names
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Generate random values for angle, start_x, start_y, end_x, and end_y for each class
objects = []
for class_name in coco_classes:
    angle = random.uniform(0, 360)  # Random angle between 0 and 360 degrees
    start_x = random.randint(10, 100)
    start_y = random.randint(10, 100)
    end_x = random.randint(150, 300)
    end_y = random.randint(150, 300)

    objects.append({
        'class': class_name,
        'angle': angle,
        'start_x': start_x,
        'start_y': start_y,
        'end_x': end_x,
        'end_y': end_y
    })

# Write the objects to objects.csv
with open('objects.csv', 'w', newline='') as csvfile:
    fieldnames = ['class', 'angle', 'start_x', 'start_y', 'end_x', 'end_y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for obj in objects:
        writer.writerow(obj)

print("objects.csv file generated with random values for all 80 COCO dataset classes.")

