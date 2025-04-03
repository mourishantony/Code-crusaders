import cv2
import torch
from ultralytics import YOLO

# Ensure YOLO uses GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 model on GPU
model = YOLO(r"source\model\best.pt")
model.to(device)

# Open webcam
cap = cv2.VideoCapture(0)  # Change to 1 if using an external webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on GPU
    results = model(frame, device=device)  

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Live Detection (GPU)", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
