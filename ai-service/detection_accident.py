from __future__ import annotations
import math
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import cv2
import numpy as np
from collections import deque
import subprocess
import os
from ultralytics import YOLO


def extract_accident_clip(video_path: str, output_dir: str, start_time: float, duration: float, clip_name: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{clip_name}.mp4")
    cmd = ["ffmpeg", "-y", "-ss", f"{start_time:.3f}", "-i", video_path, "-t", f"{duration:.3f}",
           "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", output_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path
    except:
        return None


@dataclass
class DetectionConfig:
    """Production-grade configuration with comprehensive filtering"""
    # Speed thresholds
    SPEED_DROP_MIN: float = 4.5
    SPEED_DROP_PCT: float = 0.55
    STOP_SPEED: float = 1.2
    MIN_MOVING_SPEED: float = 3.5
    MIN_PRE_COLLISION_SPEED: float = 10.0
    MAX_REALISTIC_SPEED: float = 120.0
    # IoU thresholds
    IOU_MIN: float = 0.14
    IOU_COLLISION: float = 0.28
    IOU_GROWTH_RAPID: float = 0.12
    IOU_JUMP_MIN: float = 0.18
    # Multi-indicator requirements
    MIN_INDICATORS: int = 3
    MIN_CONFIDENCE: float = 0.73
    REQUIRE_SPEED_DROP: bool = True
    REQUIRE_SUDDEN_EVENT: bool = True
    # Temporal validation
    MIN_HISTORY_FRAMES: int = 8
    SUDDEN_EVENT_WINDOW: int = 4
    # Duplicate prevention
    COOLDOWN_SEC: float = 15.0
    SKIP_FRAMES_AFTER: int = 240
    MAX_PER_PAIR: int = 1
    SPATIAL_COOLDOWN_RADIUS: float = 120.0
    # Traffic flow
    FLOW_UNIFORMITY_THRESHOLD: float = 0.25
    CONGESTION_DENSITY: int = 7
    CONGESTION_AVG_SPEED: float = 4.5
    # Trajectory
    LANE_CHANGE_LATERAL_THRESHOLD: float = 12.0
    DIRECTION_CHANGE_THRESHOLD: float = 40.0
    # Model
    MODEL_PATH: str = "yolov8n.pt"
    TRACKER: str = "bytetrack.yaml"
    CONFIDENCE: float = 0.50
    PIXEL_TO_METER: float = 0.045


def setup_logging(log_file: str = "accident_detection.log") -> logging.Logger:
    logger = logging.getLogger("AccidentDetection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w")
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


@dataclass
class VideoMetadata:
    filepath: str
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int

    def frame_to_timestamp(self, frame: int) -> float:
        return frame / self.fps

    def frame_to_timecode(self, frame: int) -> str:
        secs = self.frame_to_timestamp(frame)
        return f"{int(secs//3600):02d}:{int((secs%3600)//60):02d}:{secs%60:06.3f}"


class VideoMetadataExtractor:
    @staticmethod
    def extract(video_path: str) -> VideoMetadata:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        try:
            return VideoMetadata(
                filepath=video_path, fps=cap.get(cv2.CAP_PROP_FPS),
                total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                duration_seconds=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            cap.release()


@dataclass
class VehicleState:
    track_id: int
    frame_idx: int
    bbox: List[float]
    cx: float
    cy: float
    speed: float
    prev_speed: float
    direction: float
    prev_direction: float
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=30))
    direction_history: deque = field(default_factory=lambda: deque(maxlen=20))
    iou_history: Dict[int, deque] = field(default_factory=dict)


@dataclass
class AccidentEvent:
    detection_time: datetime
    frame_number: int
    video_timestamp: float
    timecode: str
    vehicle_ids: List[int]
    speeds_kmh: List[float]
    speed_drops_kmh: List[float]
    iou: float
    max_iou: float
    severity: str
    confidence: float
    factors: List[str]
    indicator_count: int
    clip_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.video_timestamp, "timecode": self.timecode, "frame": self.frame_number,
            "vehicles": self.vehicle_ids, "speeds_kmh": [round(s, 1) for s in self.speeds_kmh],
            "speed_drops_kmh": [round(s, 1) for s in self.speed_drops_kmh], "iou": round(self.iou, 3),
            "severity": self.severity, "confidence": round(self.confidence, 2),
            "factors": self.factors, "indicators": self.indicator_count, "clip": self.clip_path
        }

    def __str__(self) -> str:
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 ACCIDENT DETECTED - {self.severity.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️  Time: {self.timecode} (Frame {self.frame_number})
🚗 Vehicles: {self.vehicle_ids}
📊 Confidence: {self.confidence:.0%}
💥 IoU: {self.iou:.1%} (max: {self.max_iou:.1%})
🏁 Speeds: {[f"{s:.1f}" for s in self.speeds_kmh]} km/h
📉 Drops: {[f"{d:.1f}" for d in self.speed_drops_kmh]} km/h
✅ Indicators: {self.indicator_count}
⚠️  Factors: {', '.join(self.factors)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


class TrafficAnalyzer:
    """Comprehensive traffic pattern analysis"""
    
    @staticmethod
    def detect_lane_change(state: VehicleState, config: DetectionConfig) -> bool:
        """Detect lane changes based on lateral movement"""
        if len(state.position_history) < 10:
            return False
        positions = list(state.position_history)[-15:]
        if len(positions) < 10:
            return False
        
        start, end = positions[0], positions[-1]
        dx, dy = end[0] - start[0], end[1] - start[1]
        distance = math.sqrt(dx**2 + dy**2)
        if distance < 8:
            return False
        
        # Calculate max lateral deviation
        max_lateral = 0
        for pos in positions[1:-1]:
            lateral = abs((dy * (pos[0] - start[0]) - dx * (pos[1] - start[1])) / distance)
            max_lateral = max(max_lateral, lateral)
        
        return max_lateral > config.LANE_CHANGE_LATERAL_THRESHOLD
    
    @staticmethod
    def is_congestion(vehicles: List[VehicleState], config: DetectionConfig) -> bool:
        """Detect traffic congestion"""
        if len(vehicles) < config.CONGESTION_DENSITY:
            return False
        speeds = [v.speed for v in vehicles if len(v.speed_history) >= 3]
        if len(speeds) < 5:
            return False
        avg_speed = sum(speeds) / len(speeds)
        return avg_speed < config.CONGESTION_AVG_SPEED
    
    @staticmethod
    def is_uniform_flow(vehicles: List[VehicleState], config: DetectionConfig) -> bool:
        """Detect uniform traffic flow"""
        if len(vehicles) < 3:
            return False
        speeds = [v.speed for v in vehicles if len(v.speed_history) >= 5]
        if len(speeds) < 3:
            return False
        avg = sum(speeds) / len(speeds)
        if avg < 2:
            return False
        variance = sum((s - avg)**2 for s in speeds) / len(speeds)
        cv = math.sqrt(variance) / avg if avg > 0 else 1.0
        return cv < config.FLOW_UNIFORMITY_THRESHOLD


class SpeedCalculator:
    def __init__(self, metadata: VideoMetadata, config: DetectionConfig):
        self.vehicle_history: Dict[int, VehicleState] = {}
        self.metadata = metadata
        self.config = config

    def calculate(self, track_id: int, bbox: List[float], frame_idx: int) -> VehicleState:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if track_id in self.vehicle_history:
            prev = self.vehicle_history[track_id]
            dx, dy = cx - prev.cx, cy - prev.cy
            distance = math.sqrt(dx**2 + dy**2)
            speed = min(distance, self.config.MAX_REALISTIC_SPEED)
            prev_speed = prev.speed
            direction = math.degrees(math.atan2(dy, dx)) if distance > 0.5 else prev.direction
            prev_direction = prev.direction
        else:
            speed = prev_speed = direction = prev_direction = 0.0

        state = VehicleState(track_id=track_id, frame_idx=frame_idx, bbox=bbox, cx=cx, cy=cy,
                             speed=speed, prev_speed=prev_speed, direction=direction, prev_direction=prev_direction)
        state.position_history.append((cx, cy))
        state.speed_history.append(speed)
        state.direction_history.append(direction)

        if track_id in self.vehicle_history:
            state.iou_history = self.vehicle_history[track_id].iou_history

        self.vehicle_history[track_id] = state
        return state

    def to_kmh(self, pixel_speed: float) -> float:
        return pixel_speed * self.config.PIXEL_TO_METER * self.metadata.fps * 3.6

    def get_avg_speed(self, track_id: int, frames: int = 5) -> float:
        if track_id not in self.vehicle_history:
            return 0.0
        hist = list(self.vehicle_history[track_id].speed_history)[-frames:]
        return sum(hist) / len(hist) if hist else 0.0

    def get_max_recent_speed(self, track_id: int, frames: int = 10) -> float:
        if track_id not in self.vehicle_history:
            return 0.0
        hist = list(self.vehicle_history[track_id].speed_history)[-frames:]
        return max(hist) if hist else 0.0

    def is_moving(self, track_id: int) -> bool:
        return self.get_avg_speed(track_id, 5) > self.config.MIN_MOVING_SPEED

    def is_sudden_deceleration(self, track_id: int) -> Tuple[bool, float]:
        """Detect sudden vs gradual deceleration"""
        if track_id not in self.vehicle_history:
            return False, 0.0
        hist = list(self.vehicle_history[track_id].speed_history)
        if len(hist) < 8:
            return False, 0.0
        
        before = hist[-10:-3] if len(hist) >= 10 else hist[:-3]
        recent = hist[-3:]
        if not before or not recent:
            return False, 0.0
        
        avg_before = sum(before) / len(before)
        avg_recent = sum(recent) / len(recent)
        drop = avg_before - avg_recent
        drop_pct = (drop / avg_before * 100) if avg_before > 1 else 0
        
        is_sudden = drop > self.config.SPEED_DROP_MIN and drop_pct > self.config.SPEED_DROP_PCT * 100
        return is_sudden, drop_pct

    def is_gradual_slowdown(self, track_id: int) -> bool:
        """Detect gradual slowdown (normal traffic)"""
        if track_id not in self.vehicle_history:
            return False
        hist = list(self.vehicle_history[track_id].speed_history)[-15:]
        if len(hist) < 10:
            return False
        decreasing = sum(1 for i in range(1, len(hist)) if hist[i] <= hist[i-1])
        return decreasing > len(hist) * 0.7


class CollisionDetector:
    @staticmethod
    def calc_iou(box_a: List[float], box_b: List[float]) -> float:
        x_a, y_a = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        x_b, y_b = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        inter = max(0, x_b - x_a) * max(0, y_b - y_a)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def calc_approach_rate(state1: VehicleState, vid2: int) -> float:
        if vid2 not in state1.iou_history:
            return 0.0
        hist = list(state1.iou_history[vid2])[-8:]
        return (hist[-1] - hist[0]) / len(hist) if len(hist) >= 3 else 0.0


class ProductionDetector:
    """Production-grade detector with comprehensive filtering"""

    def __init__(self, config: DetectionConfig, logger: logging.Logger,
                 metadata: VideoMetadata, speed_calc: SpeedCalculator):
        self.config = config
        self.logger = logger
        self.metadata = metadata
        self.speed_calc = speed_calc
        self.collision = CollisionDetector()
        self.traffic = TrafficAnalyzer()

        self.accidents: List[AccidentEvent] = []
        self.pair_count: Dict[Tuple[int, int], int] = {}
        self.skip_until: int = 0
        self.recent_detections: deque = deque(maxlen=10)
        self.analyzed = 0
        self.rejected = 0
        self.rejection_reasons: Dict[str, int] = {}

    def should_skip(self, frame: int) -> bool:
        return frame < self.skip_until

    def _log_rejection(self, reason: str):
        self.rejected += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def _is_near_recent_detection(self, cx1: float, cy1: float, cx2: float, cy2: float) -> bool:
        center_x, center_y = (cx1 + cx2) / 2, (cy1 + cy2) / 2
        for det_x, det_y in self.recent_detections:
            if math.sqrt((center_x - det_x)**2 + (center_y - det_y)**2) < self.config.SPATIAL_COOLDOWN_RADIUS:
                return True
        return False

    def analyze_pair(self, s1: VehicleState, s2: VehicleState, iou: float,
                     frame: int, all_vehicles: List[VehicleState]) -> Optional[AccidentEvent]:

        if self.should_skip(frame):
            return None

        pair_key = tuple(sorted([s1.track_id, s2.track_id]))

        # Update IoU history
        if s2.track_id not in s1.iou_history:
            s1.iou_history[s2.track_id] = deque(maxlen=30)
        if s1.track_id not in s2.iou_history:
            s2.iou_history[s1.track_id] = deque(maxlen=30)
        s1.iou_history[s2.track_id].append(iou)
        s2.iou_history[s1.track_id].append(iou)

        # === FAST PRE-FILTERS ===
        if self.pair_count.get(pair_key, 0) >= self.config.MAX_PER_PAIR:
            return None
        if iou < self.config.IOU_MIN:
            return None
        if (len(s1.speed_history) < self.config.MIN_HISTORY_FRAMES or
            len(s2.speed_history) < self.config.MIN_HISTORY_FRAMES):
            return None
        if self._is_near_recent_detection(s1.cx, s1.cy, s2.cx, s2.cy):
            return None

        self.analyzed += 1

        # === TRAFFIC CONTEXT ===
        if self.traffic.is_congestion(all_vehicles, self.config):
            self._log_rejection("traffic_congestion")
            return None
        
        if self.traffic.is_uniform_flow(all_vehicles, self.config):
            self._log_rejection("uniform_flow")
            return None

        # === LANE CHANGE DETECTION ===
        if self.traffic.detect_lane_change(s1, self.config):
            self._log_rejection("lane_change_v1")
            return None
        if self.traffic.detect_lane_change(s2, self.config):
            self._log_rejection("lane_change_v2")
            return None

        # === SPEED CHECKS ===
        max_speed = max(self.speed_calc.get_max_recent_speed(s1.track_id),
                        self.speed_calc.get_max_recent_speed(s2.track_id))
        if max_speed < self.config.MIN_PRE_COLLISION_SPEED:
            self._log_rejection("insufficient_pre_speed")
            return None

        if self.speed_calc.is_gradual_slowdown(s1.track_id) or self.speed_calc.is_gradual_slowdown(s2.track_id):
            self._log_rejection("gradual_slowdown")
            return None

        # === VALIDATE ===
        metrics = self._calc_metrics(s1, s2, iou)
        result = self._validate(s1, s2, metrics)

        if result:
            conf, factors, ind_count = result
            if conf >= self.config.MIN_CONFIDENCE and ind_count >= self.config.MIN_INDICATORS:
                event = self._create_event(s1, s2, metrics, frame, conf, factors, ind_count)
                self.pair_count[pair_key] = self.pair_count.get(pair_key, 0) + 1
                self.skip_until = frame + self.config.SKIP_FRAMES_AFTER
                self.recent_detections.append(((s1.cx + s2.cx) / 2, (s1.cy + s2.cy) / 2))
                self.logger.info(f"⏭️  Skipping {self.config.SKIP_FRAMES_AFTER} frames")
                return event
            else:
                self._log_rejection("insufficient_confidence")

        return None

    def _calc_metrics(self, s1: VehicleState, s2: VehicleState, iou: float) -> Dict:
        drop1 = s1.prev_speed - s1.speed
        drop2 = s2.prev_speed - s2.speed
        pct1 = (drop1 / s1.prev_speed * 100) if s1.prev_speed > 1 else 0
        pct2 = (drop2 / s2.prev_speed * 100) if s2.prev_speed > 1 else 0

        iou_hist = s1.iou_history.get(s2.track_id, deque())
        max_iou = max(iou_hist) if iou_hist else iou
        growth = self.collision.calc_approach_rate(s1, s2.track_id)

        # IoU jump (sudden collision signature)
        iou_jump = 0.0
        if len(iou_hist) >= 5:
            iou_jump = iou_hist[-1] - iou_hist[-5]

        return {"drop1": drop1, "drop2": drop2, "pct1": pct1, "pct2": pct2,
                "speed1": s1.speed, "speed2": s2.speed,
                "avg1": self.speed_calc.get_avg_speed(s1.track_id),
                "avg2": self.speed_calc.get_avg_speed(s2.track_id),
                "iou": iou, "max_iou": max_iou, "growth": growth, "iou_jump": iou_jump}

    def _validate(self, s1: VehicleState, s2: VehicleState, m: Dict) -> Optional[Tuple[float, List[str], int]]:
        """Strict multi-indicator validation"""
        indicators, scores, factors = [], [], []

        # === INDICATOR 1: Sudden speed drop (REQUIRED) ===
        sudden1, pct1 = self.speed_calc.is_sudden_deceleration(s1.track_id)
        sudden2, pct2 = self.speed_calc.is_sudden_deceleration(s2.track_id)

        if not (sudden1 or sudden2):
            return None  # NO SUDDEN EVENT = NO ACCIDENT

        if sudden1:
            indicators.append("sudden_decel_v1")
            scores.append(min(1.0, pct1 / 70))
            factors.append(f"V{s1.track_id} sudden brake {pct1:.0f}%")
        if sudden2:
            indicators.append("sudden_decel_v2")
            scores.append(min(1.0, pct2 / 70))
            factors.append(f"V{s2.track_id} sudden brake {pct2:.0f}%")

        # === INDICATOR 2: IoU jump (sudden collision signature) ===
        if m["iou_jump"] > self.config.IOU_JUMP_MIN:
            indicators.append("iou_jump")
            scores.append(min(1.0, m["iou_jump"] / 0.25))
            factors.append(f"Sudden overlap increase {m['iou_jump']:.2f}")

        # === INDICATOR 3: High IoU collision ===
        if m["iou"] > self.config.IOU_COLLISION:
            indicators.append("high_iou")
            scores.append(min(1.0, m["iou"] / 0.35))
            factors.append(f"Physical overlap {m['iou']:.1%}")

        # === INDICATOR 4: Rapid approach ===
        if m["growth"] > self.config.IOU_GROWTH_RAPID:
            indicators.append("rapid_approach")
            scores.append(min(1.0, m["growth"] / 0.18))
            factors.append("Rapid approach detected")

        # === INDICATOR 5: Both stopped after moving fast ===
        both_were_fast = (m["avg1"] > self.config.MIN_PRE_COLLISION_SPEED or
                          m["avg2"] > self.config.MIN_PRE_COLLISION_SPEED)
        both_now_stopped = (m["speed1"] < self.config.STOP_SPEED and m["speed2"] < self.config.STOP_SPEED)

        if both_were_fast and both_now_stopped and m["iou"] > self.config.IOU_COLLISION:
            indicators.append("collision_stop")
            scores.append(0.95)
            factors.append("Both vehicles stopped after collision")

        # === INDICATOR 6: Extreme collision ===
        if m["iou"] > 0.35 and m["growth"] > 0.12:
            indicators.append("extreme_collision")
            scores.append(1.0)
            factors.append("Extreme collision confirmed")

        ind_count = len(set(indicators))
        if ind_count < self.config.MIN_INDICATORS:
            return None

        confidence = sum(scores) / len(scores) if scores else 0.0

        # Bonuses
        if ind_count >= 4:
            confidence = min(1.0, confidence + 0.12)
        elif ind_count >= 3:
            confidence = min(1.0, confidence + 0.08)
        if m["iou"] > 0.40:
            confidence = min(1.0, confidence + 0.10)
        if max(pct1, pct2) > 85:
            confidence = min(1.0, confidence + 0.10)

        return confidence, factors, ind_count

    def _create_event(self, s1: VehicleState, s2: VehicleState, m: Dict,
                      frame: int, conf: float, factors: List[str], ind_count: int) -> AccidentEvent:

        speed1_kmh = self.speed_calc.to_kmh(s1.speed)
        speed2_kmh = self.speed_calc.to_kmh(s2.speed)
        drop1_kmh = self.speed_calc.to_kmh(m["drop1"])
        drop2_kmh = self.speed_calc.to_kmh(m["drop2"])

        avg_speed = (speed1_kmh + speed2_kmh) / 2
        max_drop = max(m["pct1"], m["pct2"])

        if (m["iou"] > 0.35 and avg_speed > 35) or max_drop > 85:
            severity = "Critical"
        elif (m["iou"] > 0.28 and avg_speed > 20) or max_drop > 70:
            severity = "High"
        elif (m["iou"] > 0.22 and avg_speed > 12) or max_drop > 60:
            severity = "Medium"
        else:
            severity = "Low"

        return AccidentEvent(
            detection_time=datetime.now(), frame_number=frame,
            video_timestamp=self.metadata.frame_to_timestamp(frame),
            timecode=self.metadata.frame_to_timecode(frame),
            vehicle_ids=[s1.track_id, s2.track_id],
            speeds_kmh=[speed1_kmh, speed2_kmh],
            speed_drops_kmh=[drop1_kmh, drop2_kmh],
            iou=m["iou"], max_iou=m["max_iou"],
            severity=severity, confidence=conf,
            factors=factors, indicator_count=ind_count
        )

    def save_results(self, json_path: str):
        data = {
            "video": self.metadata.filepath,
            "duration_sec": round(self.metadata.duration_seconds, 2),
            "fps": self.metadata.fps,
            "analysis_date": datetime.now().isoformat(),
            "statistics": {
                "pairs_analyzed": self.analyzed,
                "candidates_rejected": self.rejected,
                "accidents_detected": len(self.accidents),
                "rejection_breakdown": self.rejection_reasons
            },
            "config": {
                "min_indicators": self.config.MIN_INDICATORS,
                "min_confidence": self.config.MIN_CONFIDENCE,
                "require_speed_drop": self.config.REQUIRE_SPEED_DROP,
            },
            "accidents": [a.to_dict() for a in self.accidents],
        }

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"✅ Saved {len(self.accidents)} accidents to {json_path}")


class AccidentDetectionSystem:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = setup_logging()
        self.logger.info(f"Loading model: {config.MODEL_PATH}")
        self.model = YOLO(config.MODEL_PATH)

    def process_video(self, video_path: str):
        self.logger.info(f"Processing: {video_path}")

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Not found: {video_path}")

        metadata = VideoMetadataExtractor.extract(video_path)
        self.logger.info(f"Video: {metadata.fps:.1f} FPS, {metadata.duration_seconds:.1f}s, {metadata.width}x{metadata.height}")

        speed_calc = SpeedCalculator(metadata, self.config)
        detector = ProductionDetector(self.config, self.logger, metadata, speed_calc)

        results = self.model.track(
            source=video_path,
            tracker=self.config.TRACKER,
            persist=True,
            stream=True,
            conf=self.config.CONFIDENCE,
            verbose=False,
        )

        frame_idx = 0
        progress_interval = int(metadata.fps * 5)

        try:
            for result in results:
                frame_idx += 1

                if result.boxes is None or result.boxes.id is None:
                    continue

                # Calculate states
                vehicles = []
                for box, track_id in zip(result.boxes.xyxy, result.boxes.id):
                    state = speed_calc.calculate(int(track_id), box.tolist(), frame_idx)
                    vehicles.append(state)

                # Progress
                if frame_idx % progress_interval == 0:
                    tc = metadata.frame_to_timecode(frame_idx)
                    self.logger.info(f"⏱️  {tc} | Vehicles: {len(vehicles)}")

                # Analyze pairs
                for v1, v2 in itertools.combinations(vehicles, 2):
                    iou = CollisionDetector.calc_iou(v1.bbox, v2.bbox)
                    event = detector.analyze_pair(v1, v2, iou, frame_idx, vehicles)

                    if event:
                        detector.accidents.append(event)
                        self.logger.info(
                            f"🚨 ACCIDENT: {event.timecode} | "
                            f"Vehicles {event.vehicle_ids} | "
                            f"Confidence {event.confidence:.0%}"
                        )
                        print(event)

        except KeyboardInterrupt:
            self.logger.warning("Interrupted")
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            raise
        finally:
            # Save results
            detector.save_results("accidents.json")

            # Generate clips
            if detector.accidents:
                self.logger.info(f"Extracting {len(detector.accidents)} clips...")
                for i, acc in enumerate(detector.accidents, 1):
                    start = max(0, acc.video_timestamp - 2.0)
                    name = f"accident_{i}_t{acc.video_timestamp:.0f}_v{acc.vehicle_ids[0]}_{acc.vehicle_ids[1]}"
                    clip_path = extract_accident_clip(video_path, "accident_clips", start, 8.0, name)
                    acc.clip_path = clip_path

                detector.save_results("accidents.json")

            # Summary with rejection breakdown
            self.logger.info(
                f"\n{'='*60}\n"
                f"ANALYSIS COMPLETE\n"
                f"{'='*60}\n"
                f"Frames: {frame_idx}\n"
                f"Duration: {metadata.duration_seconds:.1f}s\n"
                f"Pairs analyzed: {detector.analyzed}\n"
                f"Candidates rejected: {detector.rejected}\n"
                f"Accidents detected: {len(detector.accidents)}\n"
            )
            
            # Show rejection breakdown
            if detector.rejection_reasons:
                self.logger.info("\nREJECTION BREAKDOWN:")
                for reason, count in sorted(detector.rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                    self.logger.info(f"  {reason}: {count}")
            
            self.logger.info("="*60)


def main():
    print("🚗 Production-Grade Accident Detection System")
    print("=" * 60)
    print("✅ Filters: Lane changes, Congestion, Gradual slowdowns, Uniform flow")
    print("✅ Requires: 3+ indicators, Sudden events, High confidence")
    print("=" * 60 + "\n")

    config = DetectionConfig(
        # Strict speed thresholds
        SPEED_DROP_MIN=4.5,
        SPEED_DROP_PCT=0.55,
        STOP_SPEED=1.2,
        MIN_MOVING_SPEED=3.5,
        MIN_PRE_COLLISION_SPEED=10.0,
        
        # Conservative IoU thresholds
        IOU_MIN=0.14,
        IOU_COLLISION=0.28,
        IOU_GROWTH_RAPID=0.12,
        IOU_JUMP_MIN=0.18,
        
        # Strict multi-indicator requirements
        MIN_INDICATORS=3,
        MIN_CONFIDENCE=0.73,
        REQUIRE_SPEED_DROP=True,
        REQUIRE_SUDDEN_EVENT=True,
        
        # Temporal validation
        MIN_HISTORY_FRAMES=8,
        SUDDEN_EVENT_WINDOW=4,
        
        # Duplicate prevention
        SKIP_FRAMES_AFTER=240,
        COOLDOWN_SEC=15.0,
        SPATIAL_COOLDOWN_RADIUS=120.0,
        
        # Traffic flow detection
        FLOW_UNIFORMITY_THRESHOLD=0.25,
        CONGESTION_DENSITY=7,
        CONGESTION_AVG_SPEED=4.5,
        
        # Lane change detection
        LANE_CHANGE_LATERAL_THRESHOLD=12.0,
        DIRECTION_CHANGE_THRESHOLD=40.0,
    )

    system = AccidentDetectionSystem(config)

    try:
        system.process_video("public/traffice.mp4")
        print("\n✅ Analysis complete! Check accidents.json for results.")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())