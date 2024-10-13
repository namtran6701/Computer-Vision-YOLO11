# YOLO Object Detection with Enhancements

This project implements an enhanced YOLO (You Only Look Once) object detection system using Python. It supports processing video files and webcam feeds, with various customization options.

## Features

- Support for multiple YOLO model sizes (nano, small, medium, large, xlarge)
- Video file processing and webcam feed support
- Customizable confidence threshold
- Class-specific detection
- FPS (Frames Per Second) display
- Object counting
- Color-coded bounding boxes for different classes
- Command-line argument support for easy configuration

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO
- tqdm

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/yolo-object-detection.git
   cd yolo-object-detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLO model weights (if not already present in the project directory).

## Usage

Run the script using the following command:

python main.py [options]
 
 
### Command-line Options

- `--classes`: Class IDs to detect (default: detect all classes)
- `-m, --model`: Choose YOLO model size (n, s, m, l, x) (default: n)
- `-c, --confidence`: Confidence threshold (0.0 - 1.0) (default: 0.3)
- `-i, --input`: Input directory for video files (default: "input")
- `-o, --output`: Output directory for processed videos (default: "output")
- `-w, --webcam`: Use webcam instead of video files
- `-d, --device`: Webcam device number (default: 0)
- `-t, --track`: Enable object tracking (not implemented in current version)

### Examples

1. Simply use webcam with nano model 
   ```
   python main.py -w
   ```

2. Process video files in the "input" directory using the nano model:
   ```
   python main.py -m n -i input -o output
   ```

3 Use webcam with medium model and 0.5 confidence threshold:
   ```
   python main.py -m m -c 0.5 -w
   ```

4. Detect only specific classes (e.g., persons and cars) in video files:
   ```
   python main.py --classes 0 2 -i input -o output
   ```

## Project Structure

- `main.py`: The main script containing all the classes and functions
- `Config`: Dataclass for storing configuration settings
- `YOLOModel`: Class for handling YOLO model operations
- `VideoProcessor`: Class for processing videos and webcam feeds

## Output

- Processed videos are saved in the specified output directory
- Each frame displays:
  - Detected objects with bounding boxes and labels
  - Confidence scores for each detection
  - Total number of objects detected
  - Current FPS

## Notes

- Press 'q' to quit the video display window
- The script automatically downloads the YOLO model if not found in the project directory

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/yolo-object-detection/issues) if you want to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.