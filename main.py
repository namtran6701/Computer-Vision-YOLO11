import cv2
import numpy as np
import os
import sys
from typing import Tuple, List, Optional
from dataclasses import dataclass
from ultralytics import YOLO
from tqdm import tqdm
import argparse
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for the YOLO object detection script."""
    model_choice: str = 'n'
    confidence_threshold: float = 0.3
    input_directory: str = "input"
    output_directory: str = "output"
    use_webcam: bool = False
    webcam_device: int = 0
    classes: List[int] = None  # Class IDs to detect
    track_objects: bool = False  # Enable object tracking


    @classmethod
    def from_args(cls):
        """Create a Config object from command line arguments."""
        parser = argparse.ArgumentParser(description="YOLO Object Detection with Enhancements")
        parser.add_argument('--classes', type=int, nargs='+', default=None,
                            help="Class IDs to detect (default: detect all classes)")
        parser.add_argument('-m', '--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                            help="Choose YOLO model (n, s, m, l, x)")
        parser.add_argument('-c', '--confidence', type=float, default=0.3,
                            help="Confidence threshold (0.0 - 1.0)")
        parser.add_argument('-i', '--input', type=str, default="input",
                            help="Input directory for video files")
        parser.add_argument('-o', '--output', type=str, default="output",
                            help="Output directory for processed videos")
        parser.add_argument('-w', '--webcam', action='store_true',
                            help="Use webcam instead of video files")
        parser.add_argument('-d', '--device', type=int, default=0,
                            help="Webcam device number (default: 0)")
        parser.add_argument('-t', '--track', action='store_true', help="Enable object tracking")
        args = parser.parse_args()
        return cls(args.model, args.confidence, args.input, args.output, args.webcam, args.device, args.classes, args.track)

class YOLOModel:
    """Class to handle YOLO model operations."""

    def __init__(self, model_choice: str):
        self.model = self.load_model(model_choice)
        self.class_names = self.model.names
        self.color_map = self.generate_colors()

    @staticmethod
    def load_model(model_choice: str) -> YOLO:
        """Load and return the YOLO model."""
        model_path = f'yolo11{model_choice}.pt'
        if not os.path.exists(model_path):
            logger.info(f"Model file {model_path} not found. Downloading...")
            model = YOLO(f'yolo11{model_choice}.yaml')
            model = YOLO(f'yolo11{model_choice}.pt')
        else:
            model = YOLO(model_path)
        
        model.conf = 0.25  # set confidence threshold
        model.iou = 0.45  # set IOU threshold
        model.verbose = False  # set verbose to False
        return model

    def generate_colors(self):
        """Generate a random color for each class."""
        np.random.seed(42)
        colors = {}
        for i, name in enumerate(self.class_names):
            colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
        return colors

    def process_image(self, frame: np.ndarray, confidence_threshold: float, classes: List[int] = None) -> Tuple[np.ndarray, int]:
        """
        Process a single frame using the YOLO model.

        Args:
            frame: The input frame
            confidence_threshold: The confidence threshold for detections
            classes: List of class IDs to detect (None for all classes)

        Returns:
            A tuple containing the annotated frame and the number of objects detected
        """
        start_time = time.time()
        results = self.model(frame, classes=classes, verbose=False)
        detections = [(box, conf, cls) for r in results for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls) if conf >= confidence_threshold]
        
        annotated_frame = frame.copy()
        for box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box[:4])
            class_name = self.class_names[int(cls)]
            color = self.color_map[int(cls)]  # Color coding per class
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        object_count = len(detections)
        fps = 1 / (time.time() - start_time)
        cv2.putText(annotated_frame, f'Objects detected: {object_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Display FPS
        
        return annotated_frame, object_count

class VideoProcessor:
    """Class to handle video processing operations."""

    def __init__(self, model: YOLOModel, config: Config):
        self.model = model
        self.config = config

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Process a video file or webcam feed and optionally save the annotated output.

        Args:
            video_path: Path to the input video file or webcam index
            output_path: Path to save the output video (None for webcam mode)
        """
        cap = cv2.VideoCapture(video_path if isinstance(video_path, str) else self.config.webcam_device)
        
        if not cap.isOpened():
            logger.error(f"Error opening video source: {video_path}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Prepare output video writer if not in webcam mode
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                annotated_frame, object_count = self.model.process_image(frame, self.config.confidence_threshold, self.config.classes)
                                
                if out:
                    out.write(annotated_frame)
                
                cv2.imshow('YOLO Object Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

        if output_path:
            logger.info(f"Finished processing. Output video saved as: {output_path}")

    def process_all_videos(self) -> None:
        """Process all videos in the input directory or use webcam."""
        if self.config.use_webcam:
            logger.info("Starting webcam feed...")
            self.process_video(self.config.webcam_device)
        else:
            video_files = [f for f in os.listdir(self.config.input_directory) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            
            if not video_files:
                logger.warning(f"No video files found in the '{self.config.input_directory}' directory.")
                return

            os.makedirs(self.config.output_directory, exist_ok=True)

            for video_file in video_files:
                input_path = os.path.join(self.config.input_directory, video_file)
                output_path = os.path.join(self.config.output_directory, f"output_{video_file}")
                logger.info(f"Processing video: {video_file}")
                self.process_video(input_path, output_path)

def main():
    """Main function to run the YOLO object detection script."""
    config = Config.from_args()
    model = YOLOModel(config.model_choice)
    processor = VideoProcessor(model, config)
    processor.process_all_videos()

if __name__ == "__main__":
    main()
