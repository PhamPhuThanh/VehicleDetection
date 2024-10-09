from ultralytics import YOLO
import cv2
import math

model = YOLO(r"D:\Yolov8-VehicleDetection\train\weights\best.pt")

classNames = ['Car', 'Motorcycle', 'Truck', 'Bus', 'Bicycle']
class_colors = {
    'Car': (0, 255, 0),         # màu xanh lá
    'Motorcycle': (0, 0, 255),  # màu đỏ
    'Truck': (255, 0, 0),       # màu xanh dương
    'Bus': (0, 255, 255),       # màu vàng
    'Bicycle': (255, 255, 0)    # màu xanh lam
}

cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định (index 0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Doing detections using YOLOv8
    results = model(frame)

    # Once we have the results we will loop through them and we will have the bounding boxes for each of the result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            color = class_colors[class_name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()