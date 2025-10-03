from typing import Tuple
import cv2.cuda as cuda
import cv2
import time
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
import json
import os
from tkinter import messagebox

# Modules
from Modules.Annotate import Annotator
from Modules.Load_annotations import AnnotationLoader
from Modules.Capture_UI import CameraApp
from Modules.watching_image import ImageWatcher

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class SharpeyeApp(ctk.CTk):
    # Main def function / Init
    def __init__(self):
        super().__init__()

        self.title("SHARPEYE - VC")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # navbar frame
        self.navbar = ctk.CTkFrame(self, height=55, fg_color="#333")
        self.navbar.pack(side="top", fill="x", padx=10, pady=5)

        # serial number label and entry
        self.serial_label = ctk.CTkLabel(self.navbar, text="Serial Number:", font=("Arial", 14))
        self.serial_label.pack(side="left", padx=10)
        self.serial_entry = ctk.CTkEntry(self.navbar, width=280, placeholder_text="Enter serial number")
        self.serial_entry.pack(side="left", padx=10)

        self.after(3000, self.serial_entry.focus_set)

        # button for annotation
        self.annotate_btn = ctk.CTkButton(self.navbar, text="Annotate", width=100, font=("Arial", 14), command=self.run_module_action)
        self.annotate_btn.pack(side="left", padx=5)

        # button for save annotation list as dictionary
        self.save_annotation_btn = ctk.CTkButton(self.navbar, text="Save Data", width=100, font=("Arial", 14), command=self.annotate_save_btn)
        self.save_annotation_btn.pack(side="left", padx=5)

        # button for capturing 
        self.capture_btn = ctk.CTkButton(self.navbar, text="Capture", width=100, font=("Arial", 14), command=self.capture_function)
        self.capture_btn.pack(side="left", padx=5)

        # button back to live
        self.back_btn = ctk.CTkButton(self.navbar, text="Back", width=100, font=("Arial", 14), command=self.start_live_view)
        self.back_btn.place_forget()

        # status label for messages
        self.status_label = ctk.CTkLabel(self, text="", font=("Arial", 12))
        self.status_label.pack(side="bottom", pady=5)

        # video display (use Canvas for mouse events)
        self.video_canvas = ctk.CTkCanvas(self, highlightthickness=0, bg="#333")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # for module load annotations
        self.annotate_loader = AnnotationLoader()
        self.annotator = Annotator()
        self.capture = CameraApp()

        self.use_cuda = cuda.getCudaEnabledDeviceCount() > 0
        if not self.use_cuda:
            print("Warning: CUDA not available, falling back to CPU processing")

        # GStreamer pipeline
        self.DEVICE_PATH = '/dev/video0'
        self.pipeline = (
            f"v4l2src device={self.DEVICE_PATH} ! "
            "image/jpeg,width=1920,height=1080,framerate=30/1 ! "
            "jpegdec ! video/x-raw ! "  # CPU-based JPEG decoding
            "nvvidconv ! video/x-raw,format=BGRx ! "  # GPU-based format conversion
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink"
        )

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            error_msg = (f"Error: Could not open device {self.DEVICE_PATH}. "
                         "Device may be busy or pipeline is incorrect.\n"
                         "Test: gst-launch-1.0 v4l2src device=/dev/video0 ! "
                         "image/jpeg,width=1920,height=1080,framerate=30/1 ! "
                         "jpegdec ! nvvidconv ! video/x-raw,format=BGRx ! "
                         "videoconvert ! autovideosink\n"
                         "Check: lsof /dev/video0 and kill any processes using it.")
            self.video_label.configure(text=error_msg, font=("Arial", 12))
            print(error_msg)
            return
        print("Press q to QUIT.")

        # ***************************************************** #
        self.running = True
        self.screen_width = screen_width
        self.screen_height = screen_height - 80
        self.current_frame = None
        self.show_annotated = False
        self.is_annotating = False
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.screen_width = 1280
        self.screen_height = 720
        self.annotations = []
        self.update_video()
        self.roi_cache = {}
        self.roi_gpu_cache = {}
        self.matcher_cache = {}
        self.initialize()

        # **************************************************** #


    # initialize the camera for capturing
    def initialize_camera(self):
        self.capture.initialize_camera()


    # camera capture function
    def capture_function(self):
        self.stop_live_view()
        self.capture.capture_photo()
        time.sleep(0.5)
        self.back_btn.pack(side="left", padx=5)
        self.display_image()


    # display the image from watcher
    def display_image(self):
        # display the image captured
        self.img_label = ctk.CTkLabel(self, text="", fg_color="#fff")
        self.img_label.pack(padx=10, pady=10)

        # Start watching folder
        folder_path = "/home/nvidia/Main_Folder/Inspected_images/Captured_Images"
        self.watcher = ImageWatcher(folder_path, self.img_label, update_interval=1)
        self.watcher.start()


    # stop the video live viewing
    def stop_live_view(self):
        """Stop webcam live preview and remove canvas"""
        try:
            # Release webcam if running
            if hasattr(self, "cap") and self.cap.isOpened():
                self.cap.release()
                print("Live view stopped and webcam released.")

            # Remove the video canvas if it exists
            if hasattr(self, "video_canvas") and self.video_canvas.winfo_exists():
                self.video_canvas.destroy()
                print("Live view canvas removed.")

        except Exception as e:
            print(f"Error stopping live view: {e}")


    # Start the live viewing again for back button
    def start_live_view(self):
        """Start webcam live preview and remove photo label"""
        try:
            # stop the watcher if running
            if hasattr(self, "watcher") and self.watcher:
                self.watcher.stop()
                self.watcher = None
                print("Image watcher stopped.")

            # destroy the label if it exists
            if hasattr(self, "img_label") and self.img_label and self.img_label.winfo_exists():
                self.img_label.destroy()
                self.img_label = None
                print("Image label destroyed.")

            # recreate the canvas if needed
            if not hasattr(self, "video_canvas") or not self.video_canvas.winfo_exists():
                self.video_canvas = ctk.CTkCanvas(self, highlightthickness=0, bg="#333")

            self.video_canvas.pack(fill="both", expand=True, padx=10, pady=10)

            # reinitialize camera
            self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                error_msg = (
                    f"Error: Could not open device {self.DEVICE_PATH}. "
                    "Device may be busy or pipeline is incorrect.\n"
                    f"Test pipeline: gst-launch-1.0 {self.pipeline.replace('appsink', 'autovideosink')}\n"
                    "Check: lsof /dev/video0 and kill any processes using it.\n"
                    "Verify formats: v4l2-ctl --list-formats-ext -d /dev/video0"
                )
                self.status_label.configure(text=error_msg)
                print(error_msg)
                return

            # set running to True again
            self.running = True
            self.capture_btn.place_forget()

            # restart video update loop
            self.resume_video()
            print("Live view restarted.")

        except Exception as e:
            print(f"Error starting live view: {e}")


    # initialize function from modules
    def initialize(self):
        """Initialize by loading annotations."""
        self.annotate_loader.load_annotations()
        self.annotations = self.annotate_loader.annotations
        print(f"Loaded {len(self.annotations)} annotations for display")

        # Cache ROI images on GPU
        for rect in self.annotations:
            roi_file = rect.get("roi_file")
            if not roi_file:
                continue

            roi_img = cv2.imread(roi_file, cv2.IMREAD_GRAYSCALE)
            if roi_img is None:
                print(f"Warning: Could not load ROI {roi_file}")
                continue

            self.roi_cache[roi_file] = roi_img

            if self.use_cuda:
                roi_gpu = cuda.GpuMat()
                roi_gpu.upload(roi_img)
                self.roi_gpu_cache[roi_file] = roi_gpu

        print(f"Preloaded {len(self.roi_cache)} ROI images into cache.")

    
    def matched_roi_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matches = []

        try:
            if self.use_cuda:
                gpu_frame = cuda.GpuMat()
                gpu_frame.upload(gray)
            else:
                gpu_frame = None
        except cv2.error as e:
            print(f"GPU upload failed: {e}")
            return matches

        for rect in self.annotations:
            roi_file = rect.get("roi_file")
            if not roi_file or roi_file not in self.roi_cache:
                continue

            roi_img = self.roi_cache[roi_file]

            # Skip if ROI is bigger than frame
            if roi_img.shape[0] > gray.shape[0] or roi_img.shape[1] > gray.shape[1]:
                continue

            try:
                if self.use_cuda:
                    # Ensure ROI is cached on GPU
                    if roi_file not in self.roi_gpu_cache:
                        gpu_roi = cuda.GpuMat()
                        gpu_roi.upload(roi_img)
                        self.roi_gpu_cache[roi_file] = gpu_roi
                    else:
                        gpu_roi = self.roi_gpu_cache[roi_file]

                    # Ensure matcher is cached
                    if roi_file not in self.matcher_cache:
                        self.matcher_cache[roi_file] = cuda.createTemplateMatching(
                            cv2.CV_8U, cv2.TM_CCOEFF_NORMED
                        )

                    matcher = self.matcher_cache[roi_file]

                    # Run template matching
                    result = matcher.match(gpu_frame, gpu_roi)
                    result = result.download()
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

            except cv2.error as e:
                print(f"Template matching failed for {roi_file}: {e}")
                continue

            if max_val >= 0.7:
                h_roi, w_roi = roi_img.shape[:2]
                top_left = (max_loc[0], max_loc[1])
                bottom_right = (top_left[0] + w_roi, top_left[1] + h_roi)
                matches.append((top_left, bottom_right, roi_file, max_val))

        return matches


    # Display frame to see the results live viewing
    def display_frame(self, frame):
        """Display a frame with dynamic ROI template matching results."""
        if frame is None or frame.size == 0:
            return

        display = frame.copy()
        frame_height, frame_width = display.shape[:2]
        if frame_height == 0 or frame_width == 0:
            return

        print(f"Displaying frame with {len(self.annotations)} annotations")

        # Use dynamic matcher to find ROIs anywhere in the frame
        matches = self.matched_roi_frame(display)

        # Draw matches
        if matches:
            for top_left, bottom_right, roi_file, score in matches:
                display_name = os.path.splitext(os.path.basename(roi_file))[0]
                cv2.rectangle(display, top_left, bottom_right, (0, 0, 255), 2)
                cv2.putText(
                    display,
                    f"{display_name} - ({score:.2f})",
                    (top_left[0], max(0, top_left[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            print("No detected annotation!")

        # Resize for Tkinter canvas
        canvas_width = max(self.video_canvas.winfo_width(), 1)
        canvas_height = max(self.video_canvas.winfo_height(), 1)
        scale = min(self.screen_width / frame_width, self.screen_height / frame_height)
        new_width, new_height = int(frame_width * scale), int(frame_height * scale)

        if new_width > 0 and new_height > 0:
            resized_frame = cv2.resize(display, (new_width, new_height), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=imgtk,
                anchor=tk.CENTER
            )
            self.video_canvas.imgtk = imgtk

    # start drawing
    def start_drawing(self, event):
        if not self.is_annotating:
            return
        self.drawing = True

        if self.current_frame is None or self.current_frame.size == 0:
            print("Error: No valid frame to annotate")
            self.drawing = False
            return

        frame_height, frame_width = self.current_frame.shape[:2]
        canvas_width = max(self.video_canvas.winfo_width(), 1)
        canvas_height = max(self.video_canvas.winfo_height(), 1)

        scale = min(self.screen_width / frame_width, self.screen_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        offset_x = (canvas_width - new_width) // 2
        offset_y = (canvas_height - new_height) // 2

        adj_x = max(0, min(event.x - offset_x, new_width - 1))
        adj_y = max(0, min(event.y - offset_y, new_height - 1))

        scale_x = frame_width / new_width
        scale_y = frame_height / new_height
        self.start_point = (int(adj_x * scale_x), int(adj_y * scale_y))

        # Clamp to frame bounds
        self.start_point = (
            max(0, min(self.start_point[0], frame_width - 1)),
            max(0, min(self.start_point[1], frame_height - 1))
        )
        print(f"Mouse clicked at: ({event.x}, {event.y}) → start_point: {self.start_point}")

    def update_drawing(self, event):
        if not self.is_annotating or not self.drawing:
            return

        if self.current_frame is None or self.current_frame.size == 0:
            print("Error: No valid frame to annotate")
            return

        frame_height, frame_width = self.current_frame.shape[:2]
        canvas_width = max(self.video_canvas.winfo_width(), 1)
        canvas_height = max(self.video_canvas.winfo_height(), 1)

        scale = min(self.screen_width / frame_width, self.screen_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        offset_x = (canvas_width - new_width) // 2
        offset_y = (canvas_height - new_height) // 2

        adj_x = max(0, min(event.x - offset_x, new_width - 1))
        adj_y = max(0, min(event.y - offset_y, new_height - 1))

        scale_x = frame_width / new_width
        scale_y = frame_height / new_height
        self.end_point = (int(adj_x * scale_x), int(adj_y * scale_y))

        # Clamp to frame bounds
        self.end_point = (
            max(0, min(self.end_point[0], frame_width - 1)),
            max(0, min(self.end_point[1], frame_height - 1))
        )
        print(f"Mouse dragging at: ({event.x}, {event.y}) → end_point: {self.end_point}")

        # Draw temporary rectangle
        temp_frame = self.current_frame.copy()
        if self.start_point and self.end_point:
            cv2.rectangle(temp_frame, self.start_point, self.end_point, (0, 0, 255), 1)
            self.display_frame(temp_frame)

    def stop_drawing(self, event):
        if not self.is_annotating:
            return
        self.drawing = False

        if self.current_frame is None or self.current_frame.size == 0:
            print("Error: No valid frame to annotate")
            return

        frame_height, frame_width = self.current_frame.shape[:2]
        canvas_width = max(self.video_canvas.winfo_width(), 1)
        canvas_height = max(self.video_canvas.winfo_height(), 1)

        scale = min(self.screen_width / frame_width, self.screen_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        offset_x = (canvas_width - new_width) // 2
        offset_y = (canvas_height - new_height) // 2

        adj_x = max(0, min(event.x - offset_x, new_width - 1))
        adj_y = max(0, min(event.y - offset_y, new_height - 1))

        scale_x = frame_width / new_width
        scale_y = frame_height / new_height
        self.end_point = (int(adj_x * scale_x), int(adj_y * scale_y))

        # Clamp to frame bounds
        self.end_point = (
            max(0, min(self.end_point[0], frame_width - 1)),
            max(0, min(self.end_point[1], frame_height - 1))
        )
        print(f"Mouse released at: ({event.x}, {event.y}) → end_point: {self.end_point}")

        if self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            if w > 0 and h > 0:
                # Draw the rectangle on the current frame to make it persist
                cv2.rectangle(self.current_frame, self.start_point, self.end_point, (0, 0, 255), 1)

                # Save ROI
                roi = self.current_frame[y:y+h, x:x+w]
                roi_dir = os.path.join("annotation_logs", "roi_images")
                os.makedirs(roi_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                roi_filename = f"roi_{timestamp}.png"
                roi_path = os.path.join(roi_dir, roi_filename)
                cv2.imwrite(roi_path, roi)

                rect = {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "start": self.start_point,
                    "end": self.end_point,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "roi_file": roi_path
                }
                self.annotations.append(rect)
                print(f"Rectangle + ROI saved: {rect}")
            else:
                print("Warning: Invalid rectangle size (width or height <= 0), not saved")

        # Display the frame with the persisted rectangle
        self.display_frame(self.current_frame)


    def annotate_save_btn(self):
        """Save all accumulated annotations and their ROI file paths to a JSON file."""
        if not self.annotations:
            print("No annotations to save.")
            return

        # Prepare save structure
        save_data = {
            "annotations": self.annotations,  # includes roi_file
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Ensure save directory exists
        save_dir = os.path.abspath("annotation_logs")  # Use absolute path
        os.makedirs(save_dir, exist_ok=True)

        # Save to JSON with timestamp
        filename = os.path.join(save_dir, f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, "w") as f:
            json.dump(save_data, f, indent=4)
            f.flush()  # Ensure file is written

        print(f"Annotations & ROI paths saved to {filename}")
        messagebox.showinfo("Info", "Saved Successfully!")

        time.sleep(1)  # Consider removing if not needed
        self.resume_video()
        self.annotate_loader.load_annotations()
        self.annotations = self.annotate_loader.annotations
        print(f"Loaded new {len(self.annotations)} annotations for display")

        self.initialize()

        self.video_canvas.update()
        self.video_canvas.update_idletasks()


    def annotate_current_frame(self):
        try:
            print(f"Annotating frame with shape: {self.current_frame.shape}")
            print(f"Using coordinates: {self.start_point} to {self.end_point}")
            start_time = time.time()
            annotated_frame, result = self.annotator.annotate_frame(self.current_frame.copy(), self.start_point, self.end_point)
            print(f"Annotation took {time.time() - start_time:.2f} seconds")
            print(f"Annotation result: {result}")
            self.current_frame = annotated_frame
            self.show_annotated = True
            self.status_label.configure(text=f"Module Result: {result}")
            self.display_frame(self.current_frame)
            # Reset coordinates and drawing state
            self.start_point = None
            self.end_point = None
            self.drawing = False
        except Exception as e:
            error_msg = f"Module Error: {str(e)}"
            self.status_label.configure(text=error_msg)
            print(error_msg)
            self.show_annotated = False


    def run_module_action(self):
        print("Annotate button clicked")

        # reset the drawn rectangle
        self.annotations = []

        # Capture a fresh frame and pause live feed
        ret, frame = self.cap.read()
        if not ret:
            error_msg = "Error: Failed to capture frame for annotation"
            self.status_label.configure(text=error_msg)
            print(error_msg)
            return
        self.current_frame = frame
        self.is_annotating = True  # Pause live feed
        self.show_annotated = False

        print(f"Captured frame for annotation: {frame.shape}")
        self.display_frame(self.current_frame)
        self.status_label.configure(text="Click and drag to draw a rectangle")

        # Bind mouse events for drawing
        self.video_canvas.bind("<Button-1>", self.start_drawing)
        self.video_canvas.bind("<B1-Motion>", self.update_drawing)
        self.video_canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def resume_video(self):
        print("Resume button clicked")
        self.is_annotating = False
        self.show_annotated = False
        self.start_point = None
        self.end_point = None
        self.drawing = False

        # reset the drawn rectangle
        self.annotations = []

        self.status_label.configure(text="Live feed resumed")
        self.update_video()


    def update_video(self):
        if not getattr(self, "running", False) or self.is_annotating:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            self.running = False
            if self.cap:
                self.cap.release()
            return

        # Upload frame to GPU
        gpu_frame = cuda.GpuMat()
        gpu_frame.upload(frame)

        # Resize on GPU
        scale = min(self.screen_width / frame.shape[1], self.screen_height / frame.shape[0])
        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        gpu_resized = cuda.resize(gpu_frame, (new_width, new_height))

        # Download back to CPU (needed for Tkinter display)
        frame = gpu_resized.download()

        self.current_frame = frame
        self.display_frame(frame)

        # keep updating
        self.after(33, self.update_video)


    def destroy(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy()

if __name__ == "__main__":
    app = SharpeyeApp()
    app.protocol("WM_DELETE_WINDOW", app.destroy)
    app.initialize_camera()
    app.bind('<q>', lambda event: app.destroy())
    app.mainloop()