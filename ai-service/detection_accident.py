from __future__ import annotations
import math
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    """Extract video clip with proper encoding"""
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
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        output_path,
    ]

    try:
        subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return output_path
    except:
        return None


@dataclass
class DetectionConfig:
    """Production configuration with strict thresholds"""

    # Critical thresholds
    SPEED_DROP_MIN: float = 1.0  # pixels/frame
    SPEED_DROP_PCT: float = 0.30  # 30% drop required
    STOP_SPEED: float = 2.0

    IOU_MIN: float = 0.08
    IOU_COLLISION: float = 0.18
    IOU_GROWTH: float = 0.04

    # Multi-indicator requirement (CRITICAL for false positive prevention)
    MIN_INDICATORS: int = 2  # At least 3 independent signals
    MIN_CONFIDENCE: float = 0.75  # High bar
    REQUIRE_SPEED_DROP: bool = False  # Not Mandatory

    # Temporal validation
    CONFIRMATION_FRAMES: int = 3
    LOOKBACK_FRAMES: int = 10

    # Duplicate prevention
    COOLDOWN_SEC: float = 10.0
    SKIP_FRAMES_AFTER: int = 150  # ~5 seconds
    MAX_PER_PAIR: int = 1

    # Vehicle filtering
    MIN_MOVING_SPEED: float = 4.0
    MAX_REALISTIC_SPEED: float = 120.0

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

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
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
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = secs % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"


class VideoMetadataExtractor:
    @staticmethod
    def extract(video_path: str) -> VideoMetadata:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")

        try:
            return VideoMetadata(
                filepath=video_path,
                fps=cap.get(cv2.CAP_PROP_FPS),
                total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                duration_seconds=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                / cap.get(cv2.CAP_PROP_FPS),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
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
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=30))
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
            "timestamp": self.video_timestamp,
            "timecode": self.timecode,
            "frame": self.frame_number,
            "vehicles": self.vehicle_ids,
            "speeds_kmh": [round(s, 1) for s in self.speeds_kmh],
            "speed_drops_kmh": [round(s, 1) for s in self.speed_drops_kmh],
            "iou": round(self.iou, 3),
            "severity": self.severity,
            "confidence": round(self.confidence, 2),
            "factors": self.factors,
            "indicators": self.indicator_count,
            "clip": self.clip_path,
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


class SpeedCalculator:
    def __init__(self, metadata: VideoMetadata, config: DetectionConfig):
        self.vehicle_history: Dict[int, VehicleState] = {}
        self.metadata = metadata
        self.config = config

    def calculate(
        self, track_id: int, bbox: List[float], frame_idx: int
    ) -> VehicleState:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if track_id in self.vehicle_history:
            prev = self.vehicle_history[track_id]
            dx, dy = cx - prev.cx, cy - prev.cy
            distance = math.sqrt(dx**2 + dy**2)
            speed = min(distance, self.config.MAX_REALISTIC_SPEED)
            prev_speed = prev.speed
            direction = (
                math.degrees(math.atan2(dy, dx)) if distance > 0.5 else prev.direction
            )
        else:
            speed = prev_speed = direction = 0.0

        state = VehicleState(
            track_id=track_id,
            frame_idx=frame_idx,
            bbox=bbox,
            cx=cx,
            cy=cy,
            speed=speed,
            prev_speed=prev_speed,
            direction=direction,
        )

        state.position_history.append((cx, cy))
        state.speed_history.append(speed)

        if track_id in self.vehicle_history:
            state.iou_history = self.vehicle_history[track_id].iou_history

        self.vehicle_history[track_id] = state
        return state

    def to_kmh(self, pixel_speed: float) -> float:
        m_per_frame = pixel_speed * self.config.PIXEL_TO_METER
        m_per_sec = m_per_frame * self.metadata.fps
        return m_per_sec * 3.6

    def get_avg_speed(self, track_id: int, frames: int = 5) -> float:
        if track_id not in self.vehicle_history:
            return 0.0
        hist = list(self.vehicle_history[track_id].speed_history)[-frames:]
        return sum(hist) / len(hist) if hist else 0.0

    def is_moving(self, track_id: int) -> bool:
        return self.get_avg_speed(track_id, 5) > self.config.MIN_MOVING_SPEED


class CollisionDetector:
    @staticmethod
    def calc_iou(box_a: List[float], box_b: List[float]) -> float:
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter = max(0, x_b - x_a) * max(0, y_b - y_a)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def calc_approach_rate(state1: VehicleState, vid2: int) -> float:
        if vid2 not in state1.iou_history:
            return 0.0
        hist = list(state1.iou_history[vid2])[-5:]
        if len(hist) < 2:
            return 0.0
        return (hist[-1] - hist[0]) / len(hist)


class ProductionDetector:
    """Production-grade detector with strict multi-indicator validation"""

    def __init__(
        self,
        config: DetectionConfig,
        logger: logging.Logger,
        metadata: VideoMetadata,
        speed_calc: SpeedCalculator,
    ):
        self.config = config
        self.logger = logger
        self.metadata = metadata
        self.speed_calc = speed_calc
        self.collision = CollisionDetector()

        self.accidents: List[AccidentEvent] = []
        self.pair_count: Dict[Tuple[int, int], int] = {}
        self.skip_until: int = 0
        self.analyzed = 0
        self.rejected = 0

    def should_skip(self, frame: int) -> bool:
        return frame < self.skip_until

    def analyze_pair(
        self, s1: VehicleState, s2: VehicleState, iou: float, frame: int
    ) -> Optional[AccidentEvent]:

        if self.should_skip(frame):
            return None

        self.analyzed += 1
        pair_key = tuple(sorted([s1.track_id, s2.track_id]))

        # Update IoU history
        if s2.track_id not in s1.iou_history:
            s1.iou_history[s2.track_id] = deque(maxlen=30)
        if s1.track_id not in s2.iou_history:
            s2.iou_history[s1.track_id] = deque(maxlen=30)

        s1.iou_history[s2.track_id].append(iou)
        s2.iou_history[s1.track_id].append(iou)

        # Check limits
        if self.pair_count.get(pair_key, 0) >= self.config.MAX_PER_PAIR:
            return None

        # CRITICAL: Both vehicles must be moving (filters parked cars)
        if not self.speed_calc.is_moving(s1.track_id) and not self.speed_calc.is_moving(
            s2.track_id
        ):
            return None

        # Calculate metrics
        metrics = self._calc_metrics(s1, s2, iou)

        # STRICT validation with multiple indicators
        result = self._validate(s1, s2, metrics, frame)

        if result:
            conf, factors, ind_count = result

            if (
                conf >= self.config.MIN_CONFIDENCE
                and ind_count >= self.config.MIN_INDICATORS
            ):

                event = self._create_event(
                    s1, s2, metrics, frame, conf, factors, ind_count
                )

                # Update tracking
                self.pair_count[pair_key] = self.pair_count.get(pair_key, 0) + 1
                self.skip_until = frame + self.config.SKIP_FRAMES_AFTER

                self.logger.info(f"⏭️  Skipping {self.config.SKIP_FRAMES_AFTER} frames")
                return event
            else:
                self.rejected += 1

        return None

    def _calc_metrics(self, s1: VehicleState, s2: VehicleState, iou: float) -> Dict:
        drop1 = s1.prev_speed - s1.speed
        drop2 = s2.prev_speed - s2.speed
        pct1 = (drop1 / s1.prev_speed * 100) if s1.prev_speed > 1 else 0
        pct2 = (drop2 / s2.prev_speed * 100) if s2.prev_speed > 1 else 0

        iou_hist = s1.iou_history.get(s2.track_id, deque())
        max_iou = max(iou_hist) if iou_hist else iou
        growth = self.collision.calc_approach_rate(s1, s2.track_id)

        return {
            "drop1": drop1,
            "drop2": drop2,
            "pct1": pct1,
            "pct2": pct2,
            "speed1": s1.speed,
            "speed2": s2.speed,
            "avg1": self.speed_calc.get_avg_speed(s1.track_id),
            "avg2": self.speed_calc.get_avg_speed(s2.track_id),
            "iou": iou,
            "max_iou": max_iou,
            "growth": growth,
        }

    def _validate(
        self, s1: VehicleState, s2: VehicleState, m: Dict, frame: int
    ) -> Optional[Tuple[float, List[str], int]]:
        """
        CRITICAL: Multi-indicator validation
        Returns: (confidence, factors, indicator_count) or None
        """
        indicators = []
        scores = []
        factors = []

        # INDICATOR 1: Major speed drop (REQUIRED)
        has_speed_drop = False
        if (
            m["drop1"] > self.config.SPEED_DROP_MIN
            and m["pct1"] > self.config.SPEED_DROP_PCT * 100
        ):
            has_speed_drop = True
            indicators.append("speed_drop_v1")
            scores.append(min(1.0, m["pct1"] / 100))
            factors.append(f"V{s1.track_id} decel {m['pct1']:.0f}%")

        if (
            m["drop2"] > self.config.SPEED_DROP_MIN
            and m["pct2"] > self.config.SPEED_DROP_PCT * 100
        ):
            has_speed_drop = True
            indicators.append("speed_drop_v2")
            scores.append(min(1.0, m["pct2"] / 100))
            factors.append(f"V{s2.track_id} decel {m['pct2']:.0f}%")

        # if self.config.REQUIRE_SPEED_DROP and not has_speed_drop:
        #     return None

        if not has_speed_drop:
            scores.append(0.4)
            factors.append("No strong speed drop (camera limitation)")

        # INDICATOR 2: High IoU collision
        if m["iou"] > self.config.IOU_COLLISION:
            indicators.append("high_iou")
            scores.append(m["iou"] / 0.4)
            factors.append(f"Physical overlap {m['iou']:.1%}")

        # INDICATOR 3: Rapid approach
        if m["growth"] > self.config.IOU_GROWTH and m["iou"] > self.config.IOU_MIN:
            indicators.append("rapid_approach")
            scores.append(min(1.0, m["growth"] / 0.15))
            factors.append("Rapid approach detected")

        # INDICATOR 4: Both vehicles stopping
        if (
            m["speed1"] < self.config.STOP_SPEED
            and m["speed2"] < self.config.STOP_SPEED
            and m["iou"] > self.config.IOU_COLLISION
            and (m["avg1"] > 6 or m["avg2"] > 6)
        ):
            indicators.append("both_stopped")
            scores.append(0.9)
            factors.append("Both vehicles stopped")

        # INDICATOR 5: High IoU growth with moderate overlap
        if m["iou"] > 0.20 and m["growth"] > 0.08:
            indicators.append("collision_confirmed")
            scores.append(0.95)
            factors.append("Collision confirmed")

        ind_count = len(set(indicators))

        # STRICT: Need minimum indicators
        if ind_count < self.config.MIN_INDICATORS:
            return None

        # Calculate confidence
        confidence = sum(scores) / len(scores) if scores else 0.0

        # Bonus for multiple indicators
        if ind_count >= 4:
            confidence = min(1.0, confidence + 0.15)
        elif ind_count >= 3:
            confidence = min(1.0, confidence + 0.10)

        # Bonus for extreme values
        if m["iou"] > 0.35:
            confidence = min(1.0, confidence + 0.10)
        if max(m["pct1"], m["pct2"]) > 80:
            confidence = min(1.0, confidence + 0.10)

        return confidence, factors, ind_count

    def _create_event(
        self,
        s1: VehicleState,
        s2: VehicleState,
        m: Dict,
        frame: int,
        conf: float,
        factors: List[str],
        ind_count: int,
    ) -> AccidentEvent:

        speed1_kmh = self.speed_calc.to_kmh(s1.speed)
        speed2_kmh = self.speed_calc.to_kmh(s2.speed)
        drop1_kmh = self.speed_calc.to_kmh(m["drop1"])
        drop2_kmh = self.speed_calc.to_kmh(m["drop2"])

        # Severity
        avg_speed = (speed1_kmh + speed2_kmh) / 2
        max_drop = max(m["pct1"], m["pct2"])

        if (m["iou"] > 0.35 and avg_speed > 35) or max_drop > 85:
            severity = "Critical"
        elif (m["iou"] > 0.28 and avg_speed > 20) or max_drop > 70:
            severity = "High"
        elif (m["iou"] > 0.20 and avg_speed > 10) or max_drop > 60:
            severity = "Medium"
        else:
            severity = "Low"

        return AccidentEvent(
            detection_time=datetime.now(),
            frame_number=frame,
            video_timestamp=self.metadata.frame_to_timestamp(frame),
            timecode=self.metadata.frame_to_timecode(frame),
            vehicle_ids=[s1.track_id, s2.track_id],
            speeds_kmh=[speed1_kmh, speed2_kmh],
            speed_drops_kmh=[drop1_kmh, drop2_kmh],
            iou=m["iou"],
            max_iou=m["max_iou"],
            severity=severity,
            confidence=conf,
            factors=factors,
            indicator_count=ind_count,
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

        # Extract metadata
        metadata = VideoMetadataExtractor.extract(video_path)
        self.logger.info(
            f"Video: {metadata.fps:.1f} FPS, {metadata.duration_seconds:.1f}s, "
            f"{metadata.width}x{metadata.height}"
        )

        # Initialize
        speed_calc = SpeedCalculator(metadata, self.config)
        detector = ProductionDetector(self.config, self.logger, metadata, speed_calc)

        # Track
        results = self.model.track(
            source=video_path,
            tracker=self.config.TRACKER,
            persist=True,
            stream=True,
            conf=self.config.CONFIDENCE,
            verbose=False,
        )

        frame_idx = 0
        progress_interval = int(metadata.fps * 5)  # Every 5 seconds

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
                    event = detector.analyze_pair(v1, v2, iou, frame_idx)

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
                    start = max(0, acc.video_timestamp - 1.0)
                    name = f"accident_{i}_t{acc.video_timestamp:.0f}_v{acc.vehicle_ids[0]}_{acc.vehicle_ids[1]}"
                    clip_path = extract_accident_clip(
                        video_path, "accident_clips", start, 6.0, name
                    )
                    acc.clip_path = clip_path

                # Re-save with clip paths
                detector.save_results("accidents.json")

            # Summary
            self.logger.info(
                f"\n{'='*60}\n"
                f"ANALYSIS COMPLETE\n"
                f"{'='*60}\n"
                f"Frames: {frame_idx}\n"
                f"Duration: {metadata.duration_seconds:.1f}s\n"
                f"Pairs analyzed: {detector.analyzed}\n"
                f"Candidates rejected: {detector.rejected}\n"
                f"Accidents detected: {len(detector.accidents)}\n"
                f"{'='*60}"
            )


def main():
    print("🚗 Production Accident Detection System")
    print("=" * 60)

    config = DetectionConfig(
        SPEED_DROP_MIN=1.0,
        SPEED_DROP_PCT=0.30,
        IOU_MIN=0.08,
        IOU_COLLISION=0.18,
        IOU_GROWTH=0.04,
        MIN_INDICATORS=2,
        MIN_CONFIDENCE=0.55,
        REQUIRE_SPEED_DROP=False,
        MIN_MOVING_SPEED=1.5,
        SKIP_FRAMES_AFTER=120,
    )

    system = AccidentDetectionSystem(config)

    try:
        system.process_video("public/traffice.mp4")
        print("\n✅ Analysis complete!")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
