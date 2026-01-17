import cv2
import sys
from ultralytics import YOLO

if len(sys.argv) < 2:
    print("Usage: python hello.py ./sample.mp4")
    sys.exit(1)

video_path = sys.argv[1]
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit(1)

# Vehicle classes in COCO dataset: bicycle(1), car(2), motorcycle(3), bus(5), truck(7)
vehicle_classes = [1, 2, 3, 5, 7]

# Add custom logic to detect e-rickshaws based on bounding box characteristics
def classify_vehicle(box, cls, conf):
    x1, y1, x2, y2 = box.xyxy[0]
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height

    # E-rickshaws are typically wider than cars but narrower than trucks
    # Adjust these thresholds based on your video/camera setup
    if cls == 2 and 1.2 < aspect_ratio < 2.0 and conf > 0.5:  # Car with specific aspect ratio
        return 'e-rickshaw'
    elif cls == 7 and 0.8 < aspect_ratio < 1.5 and conf > 0.5:  # Truck with specific aspect ratio
        return 'e-rickshaw'
    else:
        return model.names[cls]

counts = {model.names[i]: 0 for i in vehicle_classes}
counts['e-rickshaw'] = 0
vehicle_count = 0
seen_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False, imgsz=320)  # Run inference with tracking

    for result in results:
        for box in result.boxes:
            if box.id is not None:
                id_val = int(box.id[0]) if hasattr(box.id, '__len__') else int(box.id)
                if id_val not in seen_ids:
                    seen_ids.add(id_val)
                    cls = int(box.cls)
                    print(f"New vehicle: {model.names[cls]} (class {cls})")
                    if cls in vehicle_classes:
                        cls_name = classify_vehicle(box, cls, box.conf[0])
                        counts[cls_name] += 1
                        vehicle_count += 1
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = box.conf[0]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'{cls_name} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    frame_vehicle_count = sum(1 for result in results for box in result.boxes if int(box.cls) in vehicle_classes)
    cv2.putText(frame, f'Total Unique Vehicles: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    # Display real-time table
    y_offset = 50
    for veh, cnt in counts.items():
        cv2.putText(frame, f'{veh.capitalize()}: {cnt}', (frame.shape[1] - 200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_offset += 30
    cv2.putText(frame, f'Total: {vehicle_count}', (frame.shape[1] - 200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.imshow('Vehicle Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Vehicle Detection Summary:")
print(f"{'Type':<12} {'Count':<5}")
print("-" * 18)
for veh, cnt in counts.items():
    print(f"{veh.capitalize():<12} {cnt:<5}")
print(f"{'Total':<12} {vehicle_count:<5}")