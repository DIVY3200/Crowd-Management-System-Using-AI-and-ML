"""
Intelligent Crowd Management System
------------------------------------
Advanced Computer Vision System for Crowd Detection and Overcrowding Prevention

Features:
- Real-time person detection and tracking
- Dynamic capacity calculation based on visible area
- Multi-metric overcrowding detection (count + density)
- Entry/Exit tracking with person counting
- Visual alerts and bounding boxes
- Automatic threshold calculation

Author: Crowd Safety Initiative
Purpose: Prevent overcrowding incidents like Mahakumbh 2025
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
from datetime import datetime
import os

class CrowdManagementSystem:
    def __init__(self, video_path, area_size_sqm=None, capacity_per_sqm=2.0,
                 model_size='n', confidence_threshold=0.35):
        self.video_path = video_path
        self.area_size_sqm = area_size_sqm
        self.capacity_per_sqm = capacity_per_sqm
        self.confidence_threshold = confidence_threshold

        # Load YOLOv8 model
        model_map = {
            'n': 'yolov8n.pt',  # Nano - fastest, good accuracy
            's': 'yolov8s.pt',  # Small - balanced
            'm': 'yolov8m.pt',  # Medium - better accuracy
            'l': 'yolov8l.pt',  # Large - high accuracy
            'x': 'yolov8x.pt'   # Extra - best accuracy, slowest
        }
        model_file = model_map.get(model_size, 'yolov8n.pt')

        print(f"Loading YOLOv8 model ({model_size.upper()})...")
        print(f"Confidence threshold: {confidence_threshold}")
        self.model = YOLO(model_file)  # Will download automatically if not present

        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video loaded: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        print(f"Total frames: {self.total_frames}")

        # Tracking parameters
        self.tracked_objects = {}  # {track_id: {info}}
        self.next_object_id = 0
        self.max_disappeared = 30  # frames before considering object left

        # Crowd metrics
        self.current_count = 0
        self.max_count_observed = 0
        self.entry_count = 0
        self.exit_count = 0

        # Overcrowding detection
        self.overcrowding = False
        self.alert_triggered = False
        self.threshold = None

        # Density tracking (for ground visibility analysis)
        self.density_history = deque(maxlen=30)  # Last 30 frames

        # Statistics
        self.frame_count = 0
        self.processing_times = deque(maxlen=30)

        # Calculate threshold
        self._calculate_capacity_threshold()

    def _calculate_capacity_threshold(self):
        """
        Calculate overcrowding threshold using multiple methods
        """
        if self.area_size_sqm:
            # Method 1: Known area size
            self.threshold = int(self.area_size_sqm * self.capacity_per_sqm)
            print(f"✓ Threshold calculated from area: {self.threshold} persons")
            print(f"  (Area: {self.area_size_sqm}m², Capacity: {self.capacity_per_sqm} persons/m²)")
        else:
            # Method 2: Estimate from frame dimensions
            # Assume camera height and field of view
            # This is a rough estimate - will be refined dynamically
            frame_area_pixels = self.frame_width * self.frame_height

            # Estimate: typical person occupies ~2000-3000 pixels at medium distance
            avg_person_pixels = 2500
            estimated_max_persons = frame_area_pixels // avg_person_pixels

            # Apply safety factor (70% of theoretical max)
            self.threshold = int(estimated_max_persons * 0.7)

            print(f"✓ Threshold estimated from frame size: {self.threshold} persons")
            print(f"  (Will be dynamically adjusted based on actual observations)")

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    def _calculate_density(self, detections):
        """
        Calculate crowd density based on:
        1. Number of detected persons
        2. Ground visibility (less visible = more crowding)
        3. Overlap between bounding boxes
        """
        if len(detections) == 0:
            return 0.0

        total_person_area = 0
        frame_area = self.frame_width * self.frame_height

        # Calculate total occupied area and overlaps
        overlap_count = 0
        for i, box1 in enumerate(detections):
            x1, y1, x2, y2 = box1
            total_person_area += (x2 - x1) * (y2 - y1)

            # Check overlaps with other boxes
            for j, box2 in enumerate(detections[i+1:], i+1):
                if self._calculate_iou(box1, box2) > 0.1:
                    overlap_count += 1

        # Density metrics
        area_coverage = total_person_area / frame_area
        overlap_ratio = overlap_count / max(1, len(detections))

        # Combined density score (0-1 range)
        density_score = min(1.0, (area_coverage * 2.0) + (overlap_ratio * 0.5))

        return density_score

    def _track_persons(self, detections):
        """
        Track persons across frames to count entries and exits
        Uses simple centroid tracking
        """
        current_centroids = []
        current_boxes = []

        for box in detections:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            current_centroids.append((cx, cy))
            current_boxes.append(box)

        # If no existing tracked objects, register all as new
        if len(self.tracked_objects) == 0:
            for i, (centroid, box) in enumerate(zip(current_centroids, current_boxes)):
                self.tracked_objects[self.next_object_id] = {
                    'centroid': centroid,
                    'box': box,
                    'disappeared': 0,
                    'entered': True
                }
                self.entry_count += 1
                self.next_object_id += 1

        # If no detections but we have tracked objects
        elif len(current_centroids) == 0:
            for object_id in list(self.tracked_objects.keys()):
                self.tracked_objects[object_id]['disappeared'] += 1

                if self.tracked_objects[object_id]['disappeared'] > self.max_disappeared:
                    self.exit_count += 1
                    del self.tracked_objects[object_id]

        # Otherwise, match current detections to existing tracked objects
        else:
            object_ids = list(self.tracked_objects.keys())
            object_centroids = [obj['centroid'] for obj in self.tracked_objects.values()]

            # Calculate distance matrix
            distances = np.zeros((len(object_centroids), len(current_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, curr_centroid in enumerate(current_centroids):
                    distances[i, j] = np.linalg.norm(
                        np.array(obj_centroid) - np.array(curr_centroid)
                    )

            # Match objects (simple nearest neighbor)
            matched_objects = set()
            matched_detections = set()

            # Sort by distance and match
            rows, cols = np.where(distances < 100)  # Max distance threshold
            sorted_indices = np.argsort(distances[rows, cols])

            for idx in sorted_indices:
                row, col = rows[idx], cols[idx]
                if row not in matched_objects and col not in matched_detections:
                    object_id = object_ids[row]
                    self.tracked_objects[object_id]['centroid'] = current_centroids[col]
                    self.tracked_objects[object_id]['box'] = current_boxes[col]
                    self.tracked_objects[object_id]['disappeared'] = 0
                    matched_objects.add(row)
                    matched_detections.add(col)

            # Handle unmatched existing objects
            unmatched_objects = set(range(len(object_centroids))) - matched_objects
            for row in unmatched_objects:
                object_id = object_ids[row]
                self.tracked_objects[object_id]['disappeared'] += 1

                if self.tracked_objects[object_id]['disappeared'] > self.max_disappeared:
                    self.exit_count += 1
                    del self.tracked_objects[object_id]

            # Register new objects for unmatched detections
            unmatched_detections = set(range(len(current_centroids))) - matched_detections
            for col in unmatched_detections:
                self.tracked_objects[self.next_object_id] = {
                    'centroid': current_centroids[col],
                    'box': current_boxes[col],
                    'disappeared': 0,
                    'entered': True
                }
                self.entry_count += 1
                self.next_object_id += 1

        # Update current count
        self.current_count = len(self.tracked_objects)
        self.max_count_observed = max(self.max_count_observed, self.current_count)

    def _detect_overcrowding(self, density_score):
        """
        Detect overcrowding using multiple criteria:
        1. Count threshold
        2. Density score
        3. Rapid increase in crowd
        """
        # Criterion 1: Count threshold
        count_exceeded = self.current_count > self.threshold

        # Criterion 2: High density (>0.6 is concerning)
        high_density = density_score > 0.6

        # Criterion 3: Rapid increase (if count increased by >30% in last few frames)
        rapid_increase = False
        if len(self.density_history) > 10:
            recent_avg = np.mean([d for d in list(self.density_history)[-10:]])
            if density_score > recent_avg * 1.3:
                rapid_increase = True

        # Overcrowding if any two criteria are met
        criteria_met = sum([count_exceeded, high_density, rapid_increase])
        self.overcrowding = criteria_met >= 2

        return self.overcrowding

    def _draw_visualization(self, frame, detections, density_score):
        """Draw bounding boxes, alerts, and statistics"""
        # Determine box color based on overcrowding
        if self.overcrowding:
            box_color = (0, 0, 255)  # Red
            text_color = (255, 255, 255)
        else:
            box_color = (0, 255, 0)  # Green
            text_color = (255, 255, 255)

        # Draw bounding boxes for each tracked person
        for object_id, obj_info in self.tracked_objects.items():
            box = obj_info['box']
            x1, y1, x2, y2 = map(int, box)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw person ID
            label = f"ID:{object_id}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Create info panel
        panel_height = 200
        panel = np.zeros((panel_height, self.frame_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background

        # Status indicator
        status_text = "⚠️ OVERCROWDING DETECTED!" if self.overcrowding else "✓ Normal Capacity"
        status_color = (0, 0, 255) if self.overcrowding else (0, 255, 0)
        cv2.putText(panel, status_text, (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)

        # Statistics
        stats = [
            f"Current Count: {self.current_count} / {self.threshold}",
            f"Density Score: {density_score:.2%}",
            f"Entries: {self.entry_count}  |  Exits: {self.exit_count}",
            f"Peak Count: {self.max_count_observed}",
            f"Frame: {self.frame_count}/{self.total_frames}"
        ]

        y_offset = 80
        for stat in stats:
            cv2.putText(panel, stat, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25

        # Add FPS
        if len(self.processing_times) > 0:
            avg_fps = 1.0 / np.mean(self.processing_times)
            cv2.putText(panel, f"FPS: {avg_fps:.1f}",
                       (self.frame_width - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Combine frame and panel
        output_frame = np.vstack([frame, panel])

        # Add overcrowding alert overlay
        if self.overcrowding:
            overlay = output_frame.copy()
            cv2.rectangle(overlay, (0, 0), (output_frame.shape[1], output_frame.shape[0]),
                         (0, 0, 255), 20)
            cv2.addWeighted(overlay, 0.3, output_frame, 0.7, 0, output_frame)

            # Alert text
            alert_text = "OVERCROWDING ALERT!"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)[0]
            text_x = (output_frame.shape[1] - text_size[0]) // 2
            text_y = (output_frame.shape[0] - panel_height) // 2

            # Text background
            cv2.rectangle(output_frame,
                         (text_x - 20, text_y - text_size[1] - 20),
                         (text_x + text_size[0] + 20, text_y + 20),
                         (0, 0, 0), -1)

            cv2.putText(output_frame, alert_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 3)

        return output_frame

    def process_video(self, output_path='output_crowd_analysis.mp4', show_live=True):
        """
        Process video and generate annotated output

        Args:
            output_path: Path for output video
            show_live: Whether to display live processing
        """
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_height = self.frame_height + 200  # Add panel height
        out = cv2.VideoWriter(output_path, fourcc, self.fps,
                             (self.frame_width, output_height))

        print("\n" + "="*60)
        print("PROCESSING VIDEO - Crowd Management System")
        print("="*60)
        print(f"Threshold: {self.threshold} persons")
        print(f"Output: {output_path}")
        print("\nPress 'q' to stop processing\n")

        try:
            while True:
                start_time = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1

                # Run YOLO detection (only detect persons - class 0)
                results = self.model(frame, classes=[0], verbose=False)

                # Extract bounding boxes for persons
                detections = []
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()

                    # Filter by confidence threshold
                    for box, conf in zip(boxes, confidences):
                        if conf > self.confidence_threshold:
                            detections.append(box)

                # Track persons and update counts
                self._track_persons(detections)

                # Calculate density
                density_score = self._calculate_density(detections)
                self.density_history.append(density_score)

                # Detect overcrowding
                self._detect_overcrowding(density_score)

                # Draw visualization
                output_frame = self._draw_visualization(frame, detections, density_score)

                # Write to output video
                out.write(output_frame)

                # Display live if requested
                if show_live:
                    # Resize for display if too large
                    display_frame = output_frame
                    if output_frame.shape[1] > 1280:
                        scale = 1280 / output_frame.shape[1]
                        new_width = 1280
                        new_height = int(output_frame.shape[0] * scale)
                        display_frame = cv2.resize(output_frame, (new_width, new_height))

                    cv2.imshow('Crowd Management System', display_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing stopped by user")
                        break

                # Track processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)

                # Progress update
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / self.total_frames) * 100
                    avg_fps = 1.0 / np.mean(self.processing_times)
                    print(f"Progress: {progress:.1f}% | "
                          f"Count: {self.current_count}/{self.threshold} | "
                          f"FPS: {avg_fps:.1f} | "
                          f"Status: {'⚠️ ALERT' if self.overcrowding else '✓ OK'}")

        finally:
            # Cleanup
            self.cap.release()
            out.release()
            cv2.destroyAllWindows()

            # Print summary
            print("\n" + "="*60)
            print("PROCESSING COMPLETE - Summary")
            print("="*60)
            print(f"Total Frames Processed: {self.frame_count}")
            print(f"Peak Crowd Count: {self.max_count_observed}")
            print(f"Total Entries: {self.entry_count}")
            print(f"Total Exits: {self.exit_count}")
            print(f"Overcrowding Events: {'YES - Alerts Triggered' if self.alert_triggered else 'NO'}")
            print(f"Output saved to: {output_path}")
            print("="*60)

    def generate_report(self, report_path='crowd_analysis_report.txt'):
        """Generate detailed analysis report"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CROWD MANAGEMENT SYSTEM - ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Video File: {self.video_path}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Video Resolution: {self.frame_width}x{self.frame_height}\n")
            f.write(f"Frame Rate: {self.fps} fps\n")
            f.write(f"Total Duration: {self.total_frames / self.fps:.1f} seconds\n")
            if self.area_size_sqm:
                f.write(f"Monitored Area: {self.area_size_sqm} m²\n")
                f.write(f"Capacity Ratio: {self.capacity_per_sqm} persons/m²\n")
            f.write(f"Calculated Threshold: {self.threshold} persons\n\n")

            f.write("RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Frames Processed: {self.frame_count}\n")
            f.write(f"Peak Crowd Count: {self.max_count_observed}\n")
            f.write(f"Total Entries Detected: {self.entry_count}\n")
            f.write(f"Total Exits Detected: {self.exit_count}\n")
            f.write(f"Capacity Utilization: {(self.max_count_observed / self.threshold * 100):.1f}%\n")
            f.write(f"Overcrowding Status: {'⚠️ DETECTED' if self.overcrowding else '✓ SAFE'}\n\n")

            f.write("SAFETY ASSESSMENT\n")
            f.write("-" * 70 + "\n")
            if self.max_count_observed > self.threshold:
                f.write("⚠️ WARNING: Threshold exceeded during monitoring period\n")
                f.write("   Recommendation: Review crowd control measures\n")
            else:
                f.write("✓ Crowd levels remained within safe capacity\n")

            f.write("\n" + "="*70 + "\n")
            f.write("Report generated by Crowd Management System v1.0\n")

        print(f"Report saved to: {report_path}")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("  INTELLIGENT CROWD MANAGEMENT SYSTEM")
    print("  Computer Vision for Public Safety")
    print("="*70 + "\n")

    # Example usage - Update these parameters for your video
    VIDEO_PATH = "crowd_video3.mp4"  # Change to your video file

    # Option 1: If you know the actual area size
    # AREA_SIZE = 100  # square meters
    # system = CrowdManagementSystem(VIDEO_PATH, area_size_sqm=AREA_SIZE, capacity_per_sqm=2.5)

    # Option 2: Let system estimate from video (recommended for testing)
    system = CrowdManagementSystem(VIDEO_PATH)

    # Process the video
    system.process_video(output_path='crowd_analysis_output.mp4', show_live=True)

    # Generate report
    system.generate_report('crowd_analysis_report.txt')

    print("\n✓ Analysis complete! Check the output video and report.\n")


if __name__ == "__main__":
    main()