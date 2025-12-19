from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.track(
    source="public/traffice.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    stream=True
)

for r in results:
    boxes = r.boxes
    if boxes is not None and boxes.id is not None:
        for box, track_id in zip(boxes.xyxy, boxes.id):
            print(f"Vehicle ID {int(track_id)} at {box.tolist()}")
