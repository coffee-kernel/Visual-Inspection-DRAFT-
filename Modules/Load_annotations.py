# Modules/Load_annotations.py
import os
import json
import cv2

class AnnotationLoader:
    def load_annotations(self):
        """Load all annotations from JSON files inside annotation_logs and validate ROI files."""
        self.annotations = []  # reset

        folder = "annotation_logs"
        roi_folder = os.path.join(folder, "roi_images")  # Subfolder for images
        if not os.path.exists(folder):
            print("[WARNING] No annotation_logs folder found.")
            return
        if not os.path.exists(roi_folder):
            print(f"[WARNING] No roi_images folder found at {roi_folder}.")
            return

        for file in os.listdir(folder):
            if not file.endswith(".json"):
                continue

            log_file = os.path.join(folder, file)
            try:
                with open(log_file, "r") as f:
                    data = json.load(f)

                # Handle both formats (dict with "annotations" or plain list)
                raw_annotations = []
                if isinstance(data, dict) and "annotations" in data:
                    raw_annotations = data["annotations"]
                elif isinstance(data, list):
                    raw_annotations = data
                else:
                    print(f"[WARNING] {file} has unexpected format, skipping...")
                    continue

                # Validate each annotation
                for ann in raw_annotations:
                    roi_file = ann.get("roi_file")
                    if not roi_file:
                        print(f"[WARNING] Annotation in {file} missing 'roi_file', skipping...")
                        continue

                    # Normalize roi_file path
                    # If absolute, use as is; if relative, assume it's relative to roi_images
                    if not os.path.isabs(roi_file):
                        # Remove any leading "annotation_logs/roi_images/" or "roi_images/" to avoid duplication
                        roi_file = roi_file.replace("annotation_logs/roi_images/", "").replace("roi_images/", "")
                        roi_file = os.path.join(roi_folder, roi_file)

                    # Ensure file ends with .png
                    if not roi_file.lower().endswith(".png"):
                        print(f"[WARNING] ROI file {roi_file} in {file} is not a .png file, skipping...")
                        continue

                    if not os.path.exists(roi_file):
                        print(f"[WARNING] ROI file not found: {roi_file} (from {file})")
                        continue
                    if cv2.imread(roi_file) is None:
                        print(f"[WARNING] ROI file unreadable: {roi_file} (from {file})")
                        continue

                    # Ensure annotation has required keys for display
                    if not all(k in ann for k in ("x", "y", "width", "height")):
                        print(f"[WARNING] Annotation in {file} missing required keys (x, y, width, height), skipping...")
                        continue

                    # Passed validation
                    self.annotations.append(ann)

                print(f"Loaded {file} with {len(raw_annotations)} annotations "
                      f"({len(self.annotations)} valid so far)")

            except Exception as e:
                print(f"[WARNING] Failed to load {file}: {e}")

        print(f"Total valid annotations loaded: {len(self.annotations)}")