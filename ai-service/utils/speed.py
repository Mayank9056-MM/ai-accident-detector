import math

vehicle_history = {}

def calculate_speed(track_id, bbox, frame_idx):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if track_id in vehicle_history:
        prev = vehicle_history[track_id]
        distance = math.sqrt((cx - prev["cx"])**2 + (cy - prev["cy"])**2)
        speed = distance            # pixels per frame
        prev_speed = prev["speed"]
    else:
        speed = 0
        prev_speed = 0

    vehicle_history[track_id] = {
        "cx": cx,
        "cy": cy,
        "speed": speed,
        "prev_speed": prev_speed,
        "frame": frame_idx
    }

    return speed, prev_speed
