import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

prototxtPath = "./models/mobilenet/deploy.prototxt"
weightsPath = "./models/mobilenet/mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

def detect_pedestrians(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.00392, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.1:  
            idx = int(detections[0, 0, i, 1])
            
            if idx != 15:  
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    return frame

root = tk.Tk()
root.title("Pedestrian Detection")

video_path = filedialog.askopenfilename()
cap = cv2.VideoCapture(video_path)

def update():
    ret, frame = cap.read()
    if ret:
        frame = detect_pedestrians(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        lbl.config(image=frame)
        lbl.image = frame
        lbl.after(10, update)
    else:
        cap.release()

lbl = tk.Label(root)
lbl.pack()

update()
root.mainloop()
