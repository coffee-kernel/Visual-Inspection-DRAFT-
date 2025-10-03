import cv2
import json
import os
from datetime import datetime
import numpy as np

class Annotator:
    def __init__(self, log_dir="/home/nvidia/Sharpeye_VC/annotation_logs"):
        self.log_dir = log_dir
        # Create annotation_logs folder if it doesn't exist
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            # Check if directory is writable
            if not os.access(self.log_dir, os.W_OK):
                raise PermissionError(f"Directory {self.log_dir} is not writable")
        except Exception as e:
            print(f"Error initializing log directory: {str(e)}")

    def annotate_frame(self, frame, top_left, bottom_right):
        try:
            # Validate input frame
            if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Invalid frame: Expected a BGR image (NumPy array with shape (height, width, 3))")
            
            # Validate coordinates
            height, width = frame.shape[:2]
            x1, y1 = top_left
            x2, y2 = bottom_right
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            # Ensure top_left is upper-left and bottom_right is lower-right
            top_left = (min(x1, x2), min(y1, y2))
            bottom_right = (max(x1, x2), max(y1, y2))

            # Make a copy of the frame to avoid modifying the original
            annotated_frame = frame.copy()

            # Draw rectangle
            color = (0, 0, 255)  # Red in BGR
            thickness = 2
            cv2.rectangle(annotated_frame, top_left, bottom_right, color, thickness)

            # Prepare annotation data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            annotation_data = {
                "timestamp": timestamp,
                "image_width": width,
                "image_height": height,
                "annotations": [
                    {
                        "type": "rectangle",
                        "top_left": top_left,
                        "bottom_right": bottom_right,
                        "color": "red"
                    }
                ]
            }

            # Save to JSON file
            json_filename = os.path.join(self.log_dir, f"annotation_{timestamp}.json")
            with open(json_filename, 'w') as f:
                json.dump(annotation_data, f, indent=4)
            print(f"Annotation saved to {json_filename}")  # Debug: Confirm file creation

            return annotated_frame, f"Annotation saved to {json_filename}"
        except Exception as e:
            print(f"Annotation Error: {str(e)}")  # Debug: Print error details
            return frame, f"Annotation Error: {str(e)}"