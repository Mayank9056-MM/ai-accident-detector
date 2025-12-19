from __future__ import annotations
import math
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import cv2
import numpy as np
from collections import deque
import subprocess
import os

from ultralytics import YOLO


def extract_accident_clip(
    video_path: str, output_dir: str, start_time: float, duration: float, clip_name: str
):
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{clip_name}.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        output_path,
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_path


def generate_accident_clips(
    video_path: str,
    accidents: List["AccidentEvent"],
    clip_duration_post: float = 5.0,
    clip_duration_pre: float = 1.0,
):
    clip_paths = []

    for idx, accident in enumerate(accidents, 1):
        start_time = max(0.0, accident.video_timestamp - clip_duration_pre)
        duration = clip_duration_pre + clip_duration_post

        clip_name = (
            f"accident_{idx}_"
            f"t{accident.video_timestamp:.2f}_"
            f"v{accident.vehicle_ids[0]}_{accident.vehicle_ids[1]}"
        )

        clip_path = extract_accident_clip(
            video_path=video_path,
            output_dir="accident_clips",
            start_time=start_time,
            duration=duration,
            clip_name=clip_name,
        )

        clip_paths.append(clip_path)

    return clip_paths


# ============================================================================
# CONFIGURATION
# ============================================================================


class DetectionStrategy(Enum):
    """Detection strategies for different accident scenarios"""

    SUDDEN_STOP = "sudden_stop"  # Rapid deceleration
    COLLISION_OVERLAP = "collision_overlap"  # Physical overlap
    TRAJECTORY_CHANGE = "trajectory_change"  # Sudden direction change
    COMBINED = "combined"  # Multiple indicators


@dataclass
class DetectionConfig:
    """Configuration parameters for accident detection"""

    # Strategy weights (must sum to 1.0)
    STRATEGY_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "speed_drop": 0.35,
            "iou_overlap": 0.30,
            "trajectory": 0.20,
            "temporal": 0.15,
        }
    )

    # Speed thresholds (adaptive based on vehicle history)
    SPEED_DROP_THRESHOLD_MIN: float = 3.0  # pixels/frame
    SPEED_DROP_THRESHOLD_MAX: float = 15.0
    SPEED_DROP_PERCENTAGE: float = 0.5  # 50% speed drop
    STOP_SPEED_THRESHOLD: float = 3.0

    # Collision thresholds
    IOU_THRESHOLD_MIN: float = 0.10
    IOU_THRESHOLD_COLLISION: float = 0.20  # Clear collision
    IOU_RAPID_INCREASE: float = 0.05  # Rapid approach

    # Temporal confirmation
    REQUIRED_FRAMES_MIN: int = 2
    REQUIRED_FRAMES_MAX: int = 5
    ACCIDENT_WINDOW_FRAMES: int = 10  # Look back window

    # Trajectory analysis
    DIRECTION_CHANGE_THRESHOLD: float = 45.0  # degrees
    TRAJECTORY_SMOOTHING: int = 5  # frames

    # Speed conversion
    PIXEL_TO_METER: float = 0.045
    FPS_TARGET: float = 30.0

    # Model settings
    MODEL_PATH: str = "yolov8n.pt"
    TRACKER_CONFIG: str = "bytetrack.yaml"
    CONFIDENCE_THRESHOLD: float = 0.45

    # Output settings
    LOG_FILE: str = "accident_detection.log"
    ACCIDENT_LOG: str = "accidents.json"
    ACCIDENT_REPORT: str = "accident_report.txt"
    DEBUG_OUTPUT: str = "debug_frames.json"

    # Detection settings
    COOLDOWN_SECONDS: float = 8.0
    MIN_DETECTION_CONFIDENCE: float = 0.60
    MAX_ACCIDENTS_PER_PAIR: int = 1


# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging(log_file: str = "accident_detection.log") -> logging.Logger:
    """Configure structured logging"""
    logger = logging.getLogger("AccidentDetection")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================================================
# VIDEO METADATA
# ============================================================================


@dataclass
class VideoMetadata:
    """Video file metadata"""

    filepath: str
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int
    creation_time: Optional[datetime] = None

    def frame_to_timestamp(self, frame_number: int) -> float:
        return frame_number / self.fps

    def frame_to_timecode(self, frame_number: int) -> str:
        seconds = self.frame_to_timestamp(frame_number)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def frame_to_datetime(self, frame_number: int) -> datetime:
        if self.creation_time:
            offset = timedelta(seconds=self.frame_to_timestamp(frame_number))
            return self.creation_time + offset
        return datetime.now()


class VideoMetadataExtractor:
    """Extract metadata from video files"""

    @staticmethod
    def extract(video_path: str) -> VideoMetadata:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            creation_time = None
            try:
                file_stat = Path(video_path).stat()
                creation_time = datetime.fromtimestamp(file_stat.st_mtime)
            except:
                pass

            return VideoMetadata(
                filepath=video_path,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration,
                width=width,
                height=height,
                creation_time=creation_time,
            )
        finally:
            cap.release()


# ============================================================================
# VEHICLE STATE TRACKING
# ============================================================================


@dataclass
class VehicleState:
    """Enhanced vehicle state with trajectory history"""

    track_id: int
    frame_idx: int
    timestamp: float
    bbox: List[float]
    cx: float
    cy: float
    speed: float
    prev_speed: float
    direction: float  # angle in degrees

    # History for analysis
    position_history: deque = field(default_factory=lambda: deque(maxlen=20))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=20))
    iou_history: Dict[int, deque] = field(default_factory=dict)

    def __post_init__(self):
        self.bbox_width = self.bbox[2] - self.bbox[0]
        self.bbox_height = self.bbox[3] - self.bbox[1]
        self.bbox_area = self.bbox_width * self.bbox_height


@dataclass
class AccidentEvent:
    """Comprehensive accident record"""

    # Timing
    detection_time: datetime
    frame_number: int
    video_timestamp: float
    timecode: str

    # Vehicles
    vehicle_ids: List[int]
    primary_vehicle: int
    speeds_kmh: List[float]
    speed_drops_kmh: List[float]

    # Collision metrics
    iou: float
    max_iou: float
    iou_growth_rate: float
    collision_location: Tuple[float, float]
    collision_severity: str

    # Detection details
    detection_strategy: str
    confidence_score: float
    contributing_factors: List[str]
    frame_sequence: List[int]
    clip_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "detection_time": self.detection_time.isoformat(),
            "frame_number": self.frame_number,
            "video_timestamp_seconds": round(self.video_timestamp, 3),
            "timecode": self.timecode,
            "vehicle_ids": self.vehicle_ids,
            "primary_vehicle": self.primary_vehicle,
            "speeds_kmh": [round(s, 2) for s in self.speeds_kmh],
            "speed_drops_kmh": [round(s, 2) for s in self.speed_drops_kmh],
            "collision_metrics": {
                "iou": round(self.iou, 4),
                "max_iou": round(self.max_iou, 4),
                "iou_growth_rate": round(self.iou_growth_rate, 4),
                "location": {
                    "x": round(self.collision_location[0], 2),
                    "y": round(self.collision_location[1], 2),
                },
            },
            "severity": self.collision_severity,
            "detection_strategy": self.detection_strategy,
            "confidence": round(self.confidence_score, 3),
            "contributing_factors": self.contributing_factors,
            "frame_sequence": self.frame_sequence,
            "clip_path": self.clip_path,
        }

    def to_readable_string(self) -> str:
        factors = ", ".join(self.contributing_factors)
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 ACCIDENT DETECTED - {self.collision_severity.upper()} SEVERITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 Time:          {self.detection_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}
🎬 Timecode:      {self.timecode}
📍 Frame:         {self.frame_number}
⏱️  Timestamp:    {self.video_timestamp:.3f}s

🚗 Vehicles:      {', '.join(f'ID-{vid}' for vid in self.vehicle_ids)}
🎯 Primary:       ID-{self.primary_vehicle}
🔍 Strategy:      {self.detection_strategy}
📊 Confidence:    {self.confidence_score:.1%}

💥 Impact Metrics:
   • IoU at detection:    {self.iou:.1%}
   • Max IoU observed:    {self.max_iou:.1%}
   • IoU growth rate:     {self.iou_growth_rate:.4f}/frame
   • Location (x,y):      ({self.collision_location[0]:.0f}, {self.collision_location[1]:.0f})

🏁 Vehicle Speeds:
{self._format_speeds()}

⚠️  Contributing Factors: {factors}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    def _format_speeds(self) -> str:
        lines = []
        for i, vid in enumerate(self.vehicle_ids):
            speed = self.speeds_kmh[i]
            drop = self.speed_drops_kmh[i]
            lines.append(f"   Vehicle {vid}: {speed:.1f} km/h (Δ -{drop:.1f} km/h)")
        return "\n".join(lines)


# ============================================================================
# ADVANCED SPEED & TRAJECTORY CALCULATOR
# ============================================================================


class AdvancedSpeedCalculator:
    """Enhanced speed calculation with trajectory analysis"""

    def __init__(self, video_metadata: VideoMetadata, pixel_to_meter: float):
        self.vehicle_history: Dict[int, VehicleState] = {}
        self.metadata = video_metadata
        self.pixel_to_meter = pixel_to_meter

    def calculate(
        self, track_id: int, bbox: List[float], frame_idx: int
    ) -> VehicleState:
        """Calculate comprehensive vehicle state"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        timestamp = self.metadata.frame_to_timestamp(frame_idx)

        if track_id in self.vehicle_history:
            prev_state = self.vehicle_history[track_id]

            # Calculate speed
            dx = cx - prev_state.cx
            dy = cy - prev_state.cy
            distance = math.sqrt(dx**2 + dy**2)
            speed = distance
            prev_speed = prev_state.speed

            # Calculate direction (angle)
            direction = (
                math.degrees(math.atan2(dy, dx))
                if distance > 0.5
                else prev_state.direction
            )
        else:
            speed = 0.0
            prev_speed = 0.0
            direction = 0.0

        # Create new state
        state = VehicleState(
            track_id=track_id,
            frame_idx=frame_idx,
            timestamp=timestamp,
            bbox=bbox,
            cx=cx,
            cy=cy,
            speed=speed,
            prev_speed=prev_speed,
            direction=direction,
        )

        # Update history
        state.position_history.append((cx, cy))
        state.speed_history.append(speed)

        # Preserve IoU history from previous state
        if track_id in self.vehicle_history:
            state.iou_history = self.vehicle_history[track_id].iou_history

        self.vehicle_history[track_id] = state
        return state

    def pixel_speed_to_kmh(self, pixel_speed: float) -> float:
        """Convert pixel/frame speed to km/h"""
        meters_per_frame = pixel_speed * self.pixel_to_meter
        meters_per_second = meters_per_frame * self.metadata.fps
        return meters_per_second * 3.6

    def get_average_speed(self, track_id: int, frames: int = 5) -> float:
        """Get smoothed average speed"""
        if track_id not in self.vehicle_history:
            return 0.0
        state = self.vehicle_history[track_id]
        if len(state.speed_history) == 0:
            return 0.0
        recent = list(state.speed_history)[-frames:]
        return sum(recent) / len(recent)

    def get_direction_change(self, track_id: int, frames: int = 3) -> float:
        """Calculate direction change over last N frames"""
        if track_id not in self.vehicle_history:
            return 0.0

        state = self.vehicle_history[track_id]
        if len(state.position_history) < frames:
            return 0.0

        positions = list(state.position_history)[-frames:]
        if len(positions) < 2:
            return 0.0

        # Calculate angle change
        dx1 = positions[1][0] - positions[0][0]
        dy1 = positions[1][1] - positions[0][1]
        dx2 = positions[-1][0] - positions[-2][0]
        dy2 = positions[-1][1] - positions[-2][1]

        if abs(dx1) < 0.1 and abs(dy1) < 0.1:
            return 0.0
        if abs(dx2) < 0.1 and abs(dy2) < 0.1:
            return 0.0

        angle1 = math.degrees(math.atan2(dy1, dx1))
        angle2 = math.degrees(math.atan2(dy2, dx2))

        diff = abs(angle2 - angle1)
        if diff > 180:
            diff = 360 - diff

        return diff

    def clear_old_tracks(self, current_frame: int, max_age: int = 300):
        """Remove stale tracks"""
        to_remove = [
            tid
            for tid, state in self.vehicle_history.items()
            if current_frame - state.frame_idx > max_age
        ]
        for tid in to_remove:
            del self.vehicle_history[tid]


# ============================================================================
# COLLISION DETECTOR WITH ADVANCED METRICS
# ============================================================================


class AdvancedCollisionDetector:
    """Enhanced collision detection with temporal analysis"""

    @staticmethod
    def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
        """Calculate Intersection over Union"""
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = box_a_area + box_b_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def calculate_distance(state1: VehicleState, state2: VehicleState) -> float:
        """Calculate center distance between vehicles"""
        return math.sqrt((state1.cx - state2.cx) ** 2 + (state1.cy - state2.cy) ** 2)

    @staticmethod
    def calculate_approach_rate(
        state1: VehicleState, state2: VehicleState, vehicle_id1: int, vehicle_id2: int
    ) -> float:
        """Calculate rate of approach (IoU growth)"""
        # Get IoU history
        if vehicle_id2 not in state1.iou_history:
            return 0.0

        iou_hist = state1.iou_history[vehicle_id2]
        if len(iou_hist) < 2:
            return 0.0

        recent = list(iou_hist)[-5:]
        if len(recent) < 2:
            return 0.0

        # Calculate growth rate
        return (recent[-1] - recent[0]) / len(recent)


# ============================================================================
# MULTI-STRATEGY ACCIDENT DETECTOR
# ============================================================================


class MultiStrategyAccidentDetector:
    """Advanced accident detection with multiple strategies"""

    def __init__(
        self,
        config: DetectionConfig,
        logger: logging.Logger,
        video_metadata: VideoMetadata,
        speed_calc: AdvancedSpeedCalculator,
    ):
        self.config = config
        self.logger = logger
        self.metadata = video_metadata
        self.speed_calc = speed_calc
        self.collision_det = AdvancedCollisionDetector()

        # Detection state
        self.detected_accidents: List[AccidentEvent] = []
        self.accident_candidates: Dict[Tuple[int, int], Dict] = {}
        self.last_accident_time: Dict[int, float] = {}
        self.accident_pair_count: Dict[Tuple[int, int], int] = {}
        self.debug_data: List[Dict] = []

    def analyze_pair(
        self, state1: VehicleState, state2: VehicleState, iou: float, frame_idx: int
    ) -> Optional[AccidentEvent]:
        """Analyze vehicle pair for accident indicators"""

        # Update IoU history
        pair_key = tuple(sorted([state1.track_id, state2.track_id]))

        if state2.track_id not in state1.iou_history:
            state1.iou_history[state2.track_id] = deque(maxlen=20)
        if state1.track_id not in state2.iou_history:
            state2.iou_history[state1.track_id] = deque(maxlen=20)

        state1.iou_history[state2.track_id].append(iou)
        state2.iou_history[state1.track_id].append(iou)

        # Check cooldown
        current_time = self.metadata.frame_to_timestamp(frame_idx)
        for vid in pair_key:
            if vid in self.last_accident_time:
                if (
                    current_time - self.last_accident_time[vid]
                    < self.config.COOLDOWN_SECONDS
                ):
                    return None

        # Check max accidents per pair
        if (
            self.accident_pair_count.get(pair_key, 0)
            >= self.config.MAX_ACCIDENTS_PER_PAIR
        ):
            return None

        # Calculate metrics
        metrics = self._calculate_metrics(state1, state2, iou, frame_idx)

        # Multi-strategy detection
        detection_result = self._multi_strategy_detection(
            state1, state2, metrics, frame_idx
        )

        if detection_result:
            strategy, confidence, factors = detection_result

            if confidence >= self.config.MIN_DETECTION_CONFIDENCE:
                event = self._create_accident_event(
                    state1,
                    state2,
                    iou,
                    metrics,
                    frame_idx,
                    strategy,
                    confidence,
                    factors,
                )

                # Update tracking
                for vid in pair_key:
                    self.last_accident_time[vid] = current_time
                self.accident_pair_count[pair_key] = (
                    self.accident_pair_count.get(pair_key, 0) + 1
                )

                return event

        return None

    def _calculate_metrics(
        self, state1: VehicleState, state2: VehicleState, iou: float, frame_idx: int
    ) -> Dict:
        """Calculate comprehensive metrics"""

        # Speed metrics
        speed_drop1 = state1.prev_speed - state1.speed
        speed_drop2 = state2.prev_speed - state2.speed
        speed_drop_pct1 = (
            (speed_drop1 / state1.prev_speed * 100) if state1.prev_speed > 1 else 0
        )
        speed_drop_pct2 = (
            (speed_drop2 / state2.prev_speed * 100) if state2.prev_speed > 1 else 0
        )

        # Average speeds
        avg_speed1 = self.speed_calc.get_average_speed(state1.track_id, 5)
        avg_speed2 = self.speed_calc.get_average_speed(state2.track_id, 5)

        # IoU metrics
        iou_hist1 = state1.iou_history.get(state2.track_id, deque())
        max_iou = max(iou_hist1) if iou_hist1 else iou
        iou_growth = self.collision_det.calculate_approach_rate(
            state1, state2, state1.track_id, state2.track_id
        )

        # Trajectory metrics
        dir_change1 = self.speed_calc.get_direction_change(state1.track_id)
        dir_change2 = self.speed_calc.get_direction_change(state2.track_id)

        # Distance
        distance = self.collision_det.calculate_distance(state1, state2)

        return {
            "speed_drop1": speed_drop1,
            "speed_drop2": speed_drop2,
            "speed_drop_pct1": speed_drop_pct1,
            "speed_drop_pct2": speed_drop_pct2,
            "avg_speed1": avg_speed1,
            "avg_speed2": avg_speed2,
            "current_speed1": state1.speed,
            "current_speed2": state2.speed,
            "iou": iou,
            "max_iou": max_iou,
            "iou_growth": iou_growth,
            "dir_change1": dir_change1,
            "dir_change2": dir_change2,
            "distance": distance,
        }

    def _multi_strategy_detection(
        self, state1: VehicleState, state2: VehicleState, metrics: Dict, frame_idx: int
    ) -> Optional[Tuple[str, float, List[str]]]:
        """Apply multiple detection strategies"""

        scores = {}
        factors = []

        # Strategy 1: Sudden Stop with Overlap
        if (
            metrics["speed_drop1"] > self.config.SPEED_DROP_THRESHOLD_MIN
            and metrics["iou"] > self.config.IOU_THRESHOLD_MIN
            and metrics["current_speed1"] < self.config.STOP_SPEED_THRESHOLD
        ):

            score = min(
                1.0,
                (metrics["speed_drop1"] / self.config.SPEED_DROP_THRESHOLD_MAX) * 0.6
                + (metrics["iou"] / self.config.IOU_THRESHOLD_COLLISION) * 0.4,
            )
            scores["sudden_stop"] = score
            factors.append(f"Sudden stop (V{state1.track_id})")

        # Strategy 2: High IoU with Speed Drop
        if metrics["iou"] > self.config.IOU_THRESHOLD_COLLISION and (
            metrics["speed_drop1"] > 2 or metrics["speed_drop2"] > 2
        ):

            score = min(
                1.0,
                (metrics["iou"] / 0.4) * 0.5
                + (max(metrics["speed_drop1"], metrics["speed_drop2"]) / 10) * 0.5,
            )
            scores["collision_overlap"] = score
            factors.append("Physical collision detected")

        # Strategy 3: Rapid IoU Increase
        if (
            metrics["iou_growth"] > self.config.IOU_RAPID_INCREASE
            and metrics["iou"] > self.config.IOU_THRESHOLD_MIN
        ):

            score = min(
                1.0, (metrics["iou_growth"] / 0.1) * 0.6 + (metrics["iou"] / 0.3) * 0.4
            )
            scores["rapid_approach"] = score
            factors.append("Rapid approach detected")

        # Strategy 4: Percentage Speed Drop
        if metrics["speed_drop_pct1"] > 50 and metrics["iou"] > 0.08:
            score = min(
                1.0,
                (metrics["speed_drop_pct1"] / 100) * 0.7 + (metrics["iou"] / 0.2) * 0.3,
            )
            scores["percentage_drop"] = score
            factors.append(f"Major deceleration ({metrics['speed_drop_pct1']:.0f}%)")

        # Strategy 5: Trajectory Change with Overlap
        if (
            metrics["dir_change1"] > self.config.DIRECTION_CHANGE_THRESHOLD
            and metrics["iou"] > 0.10
            and metrics["speed_drop1"] > 1
        ):

            score = min(
                1.0,
                (metrics["dir_change1"] / 90) * 0.5
                + (metrics["iou"] / 0.25) * 0.3
                + (metrics["speed_drop1"] / 8) * 0.2,
            )
            scores["trajectory_change"] = score
            factors.append("Trajectory change detected")

        # Strategy 6: Both vehicles stopping
        if (
            metrics["current_speed1"] < self.config.STOP_SPEED_THRESHOLD
            and metrics["current_speed2"] < self.config.STOP_SPEED_THRESHOLD
            and metrics["iou"] > self.config.IOU_THRESHOLD_COLLISION
            and (metrics["avg_speed1"] > 5 or metrics["avg_speed2"] > 5)
        ):

            score = min(
                1.0,
                (metrics["iou"] / 0.3) * 0.6
                + ((metrics["avg_speed1"] + metrics["avg_speed2"]) / 20) * 0.4,
            )
            scores["both_stopped"] = score
            factors.append("Both vehicles stopped")

        if not scores:
            return None

        # Combined confidence score
        max_strategy = max(scores.items(), key=lambda x: x[1])
        strategy_name = max_strategy[0]
        base_confidence = max_strategy[1]

        # Bonus for multiple indicators
        if len(scores) > 1:
            base_confidence = min(1.0, base_confidence + 0.1 * (len(scores) - 1))

        # High IoU bonus
        if metrics["iou"] > 0.25:
            base_confidence = min(1.0, base_confidence + 0.1)

        return strategy_name, base_confidence, factors

    def _create_accident_event(
        self,
        state1: VehicleState,
        state2: VehicleState,
        iou: float,
        metrics: Dict,
        frame_idx: int,
        strategy: str,
        confidence: float,
        factors: List[str],
    ) -> AccidentEvent:
        """Create detailed accident event"""

        # Calculate speeds in km/h
        speed1_kmh = self.speed_calc.pixel_speed_to_kmh(state1.speed)
        speed2_kmh = self.speed_calc.pixel_speed_to_kmh(state2.speed)
        drop1_kmh = self.speed_calc.pixel_speed_to_kmh(metrics["speed_drop1"])
        drop2_kmh = self.speed_calc.pixel_speed_to_kmh(metrics["speed_drop2"])

        # Determine severity
        severity = self._calculate_severity(metrics, speed1_kmh, speed2_kmh)

        # Primary vehicle (one with larger speed drop)
        primary = (
            state1.track_id
            if metrics["speed_drop1"] > metrics["speed_drop2"]
            else state2.track_id
        )

        # Frame sequence
        frame_sequence = list(range(max(1, frame_idx - 5), frame_idx + 1))

        return AccidentEvent(
            detection_time=self.metadata.frame_to_datetime(frame_idx),
            frame_number=frame_idx,
            video_timestamp=self.metadata.frame_to_timestamp(frame_idx),
            timecode=self.metadata.frame_to_timecode(frame_idx),
            vehicle_ids=[state1.track_id, state2.track_id],
            primary_vehicle=primary,
            speeds_kmh=[speed1_kmh, speed2_kmh],
            speed_drops_kmh=[drop1_kmh, drop2_kmh],
            iou=iou,
            max_iou=metrics["max_iou"],
            iou_growth_rate=metrics["iou_growth"],
            collision_location=(state1.cx, state1.cy),
            collision_severity=severity,
            detection_strategy=strategy,
            confidence_score=confidence,
            contributing_factors=factors,
            frame_sequence=frame_sequence,
        )

    def _calculate_severity(
        self, metrics: Dict, speed1_kmh: float, speed2_kmh: float
    ) -> str:
        """Calculate collision severity"""
        avg_speed = (speed1_kmh + speed2_kmh) / 2
        max_drop = max(metrics["speed_drop_pct1"], metrics["speed_drop_pct2"])

        if (metrics["iou"] > 0.3 and avg_speed > 30) or max_drop > 80:
            return "High"
        elif (metrics["iou"] > 0.2 and avg_speed > 15) or max_drop > 60:
            return "Medium"
        else:
            return "Low"

    def log_accident(self, event: AccidentEvent):
        """Record accident"""
        self.detected_accidents.append(event)
        self.logger.info(
            f"ACCIDENT at {event.timecode} | Vehicles: {event.vehicle_ids} | "
            f"Strategy: {event.detection_strategy} | Confidence: {event.confidence_score:.1%}"
        )

    def save_results(self, json_path: str, report_path: str):
        """Save results"""
        # JSON
        try:
            with open(json_path, "w") as f:
                json.dump(
                    {
                        "video_info": {
                            "filepath": self.metadata.filepath,
                            "fps": self.metadata.fps,
                            "duration": self.metadata.duration_seconds,
                            "resolution": f"{self.metadata.width}x{self.metadata.height}",
                        },
                        "detection_config": {
                            "strategies": list(self.config.STRATEGY_WEIGHTS.keys()),
                            "min_confidence": self.config.MIN_DETECTION_CONFIDENCE,
                            "cooldown_seconds": self.config.COOLDOWN_SECONDS,
                        },
                        "summary": {
                            "total_accidents": len(self.detected_accidents),
                            "analysis_date": datetime.now().isoformat(),
                        },
                        "accidents": [acc.to_dict() for acc in self.detected_accidents],
                    },
                    f,
                    indent=2,
                )
            self.logger.info(
                f"Saved {len(self.detected_accidents)} accidents to {json_path}"
            )
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {e}")

        # Report
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("COMPREHENSIVE ACCIDENT DETECTION REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Video: {self.metadata.filepath}\n")
                f.write(f"Duration: {self.metadata.duration_seconds:.2f}s\n")
                f.write(f"FPS: {self.metadata.fps:.2f}\n")
                f.write(f"Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Accidents: {len(self.detected_accidents)}\n\n")

                for i, acc in enumerate(self.detected_accidents, 1):
                    f.write(f"ACCIDENT #{i}\n")
                    f.write(acc.to_readable_string())
                    f.write("\n")

            self.logger.info(f"Saved report to {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")


# ============================================================================
# MAIN DETECTION SYSTEM
# ============================================================================


class AccidentDetectionSystem:
    """Main orchestration system"""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_FILE)
        self.video_metadata: Optional[VideoMetadata] = None

        self.logger.info(f"Loading YOLO model: {config.MODEL_PATH}")
        self.model = YOLO(config.MODEL_PATH)

    def process_video(self, video_path: str, show_progress: bool = True):
        """Process video for accident detection"""
        self.logger.info(f"Starting accident detection: {video_path}")

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Extract metadata
        self.logger.info("Extracting video metadata...")
        self.video_metadata = VideoMetadataExtractor.extract(video_path)
        self.logger.info(
            f"Video: {self.video_metadata.fps:.2f} FPS, "
            f"{self.video_metadata.duration_seconds:.2f}s, "
            f"{self.video_metadata.width}x{self.video_metadata.height}"
        )

        # Initialize components
        speed_calc = AdvancedSpeedCalculator(
            self.video_metadata, self.config.PIXEL_TO_METER
        )
        detector = MultiStrategyAccidentDetector(
            self.config, self.logger, self.video_metadata, speed_calc
        )

        # Start tracking
        results = self.model.track(
            source=video_path,
            tracker=self.config.TRACKER_CONFIG,
            persist=True,
            stream=True,
            conf=self.config.CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        frame_idx = 0
        actual_frames = 0
        progress_interval = int(self.video_metadata.fps)

        try:
            for result in results:
                frame_idx += 1
                actual_frames += 1

                if result.boxes is None or result.boxes.id is None:
                    continue

                # Process vehicles
                vehicles = []
                for box, track_id in zip(result.boxes.xyxy, result.boxes.id):
                    track_id = int(track_id)
                    bbox = box.tolist()
                    state = speed_calc.calculate(track_id, bbox, frame_idx)
                    vehicles.append(state)

                # Progress
                if show_progress and frame_idx % progress_interval == 0:
                    timecode = self.video_metadata.frame_to_timecode(frame_idx)
                    self.logger.info(
                        f"Processing: {timecode} "
                        f"({frame_idx}/{self.video_metadata.total_frames}) "
                        f"| Vehicles: {len(vehicles)}"
                    )

                # Analyze all pairs
                for v1, v2 in itertools.combinations(vehicles, 2):
                    iou = AdvancedCollisionDetector.calculate_iou(v1.bbox, v2.bbox)

                    event = detector.analyze_pair(v1, v2, iou, frame_idx)

                    if event:
                        detector.log_accident(event)
                        print(event.to_readable_string())

                # Cleanup
                if frame_idx % 100 == 0:
                    speed_calc.clear_old_tracks(frame_idx)

        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            raise
        finally:
            actual_duration = actual_frames / self.video_metadata.fps

            detector.save_results(self.config.ACCIDENT_LOG, self.config.ACCIDENT_REPORT)

            clip_paths = generate_accident_clips(
                video_path=self.video_metadata.filepath,
                accidents=detector.detected_accidents,
                clip_duration_pre=1.0,
                clip_duration_post=5.0,
            )

            for acc, path in zip(detector.detected_accidents, clip_paths):
                acc.clip_path = path

            detector.save_results(self.config.ACCIDENT_LOG, self.config.ACCIDENT_REPORT)

            self.logger.info(
                f"\n{'='*80}\n"
                f"PROCESSING COMPLETE\n"
                f"{'='*80}\n"
                f"Frames Processed: {actual_frames}\n"
                f"Video Duration: {actual_duration:.2f}s\n"
                f"FPS: {self.video_metadata.fps:.2f}\n"
                f"Accidents Detected: {len(detector.detected_accidents)}\n"
                f"Output Files:\n"
                f"  • JSON: {self.config.ACCIDENT_LOG}\n"
                f"  • Report: {self.config.ACCIDENT_REPORT}\n"
                f"  • Log: {self.config.LOG_FILE}\n"
                f"{'='*80}"
            )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point"""
    print("🚗 Advanced Accident Detection System v3.0")
    print("=" * 60)

    config = DetectionConfig(
        MODEL_PATH="yolov8n.pt",
        TRACKER_CONFIG="bytetrack.yaml",
        SPEED_DROP_THRESHOLD_MIN=3.0,
        SPEED_DROP_THRESHOLD_MAX=15.0,
        SPEED_DROP_PERCENTAGE=0.5,
        STOP_SPEED_THRESHOLD=3.0,
        IOU_THRESHOLD_MIN=0.10,
        IOU_THRESHOLD_COLLISION=0.20,
        IOU_RAPID_INCREASE=0.05,
        REQUIRED_FRAMES_MIN=2,
        REQUIRED_FRAMES_MAX=5,
        DIRECTION_CHANGE_THRESHOLD=45.0,
        PIXEL_TO_METER=0.045,
        CONFIDENCE_THRESHOLD=0.45,
        COOLDOWN_SECONDS=8.0,
        MIN_DETECTION_CONFIDENCE=0.60,
        MAX_ACCIDENTS_PER_PAIR=1,
    )

    system = AccidentDetectionSystem(config)
    video_path = "public/traffice.mp4"

    try:
        system.process_video(video_path, show_progress=True)
        print("\n✅ Analysis complete! Check output files for results.")
        return 0
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
