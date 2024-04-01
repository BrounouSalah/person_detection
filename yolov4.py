import cv2
import dlib
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

labelsPath = "./models/yolov4/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = "./models/yolov4/yolov4-tiny.weights"
configPath = "./models/yolov4/yolov4-tiny.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

trackers = []
frameIndex = 0

def detect_people(frame):
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layer_indices = net.getUnconnectedOutLayers().flatten()
    output_layers = [layer_names[i - 1] for i in output_layer_indices]
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    for out in detections:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.3 and LABELS[classID] == "person":
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.4)
    
    people_count = 0  
    
    if len(indices) > 0:  
        for i in indices.flatten():  
            box = boxes[i]
            (x, y, w, h) = box
            color = [int(c) for c in COLORS[classID % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"Person: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            people_count += 1  

    return frame, people_count


root = tk.Tk()
root.title("Person Detection and Tracking")


video_path = filedialog.askopenfilename()
cap = cv2.VideoCapture(video_path)

def update():
    ret, frame = cap.read()
    if ret:
        frame, count = detect_people(frame)  
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        lbl.config(image=frame)
        lbl.image = frame
        lbl.after(10, update)
        count_var.set(f"Detected People: {count}")  
    else:
        cap.release()


lbl = tk.Label(root)
lbl.pack()
count_var = tk.StringVar()
count_label = tk.Label(root, textvariable=count_var)
count_label.pack()

update()
root.mainloop()
