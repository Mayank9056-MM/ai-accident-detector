from ultralytics import YOLO
from utils.speed import calculate_speed, vehicle_history
from utils.collision import calculate_iou
from utils.accident_logic import is_accident
import itertools

model = YOLO("yolov8n.pt")

video_url = "public/traffice.mp4"

results = model.track(
    source=video_url,
    tracker="bytetrack.yaml",
    persist=True,
    stream=True
)

frame_idx = 0

for r in results:
    frame_idx += 1

    if r.boxes is None or r.boxes.id is None:
        continue

    vehicles = []

    for box, track_id in zip(r.boxes.xyxy, r.boxes.id):
        track_id = int(track_id)
        bbox = box.tolist()

        speed, prev_speed = calculate_speed(track_id, bbox, frame_idx)

        vehicles.append({
            "id": track_id,
            "bbox": bbox,
            "speed": speed,
            "prev_speed": prev_speed
        })

        # 🔍 DEBUG: per vehicle
        print(
            f"[Frame {frame_idx}] "
            f"Vehicle {track_id} | "
            f"Speed: {prev_speed:.2f} → {speed:.2f}"
        )

    # 🔥 collision + accident check
    for v1, v2 in itertools.combinations(vehicles, 2):
        iou = calculate_iou(v1["bbox"], v2["bbox"])

        if iou > 0.05:  # print only nearby vehicles
            print(
                f"   🔸 Checking V{v1['id']} & V{v2['id']} | IoU={iou:.2f}"
            )

        if is_accident(v1["prev_speed"], v1["speed"], iou):
            print("🚨 ACCIDENT DETECTED (Vehicle", v1["id"], ")")

        if is_accident(v2["prev_speed"], v2["speed"], iou):
            print("🚨 ACCIDENT DETECTED (Vehicle", v2["id"], ")")
