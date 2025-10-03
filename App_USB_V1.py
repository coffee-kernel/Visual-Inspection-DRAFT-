from typing import Tuple
import cv2
import time
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
import json
import os

# Modules
from Modules.Annotate import Annotator

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class SharpeyeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SHARPEYE - VC")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # navbar frame
        self.navbar = ctk.CTkFrame(self, height=50)
        self.navbar.pack(side="top", fill="x", padx=10, pady=5)

        # serial number label and entry
        self.serial_label = ctk.CTkLabel(self.navbar, text="Serial Number:", font=("Arial", 14))
        self.serial_label.pack(side="left", padx=10)
        self.serial_entry = ctk.CTkEntry(self.navbar, width=280, placeholder_text="Enter serial number")
        self.serial_entry.pack(side="left", padx=10)

        # button for annotation
        self.annotate_btn = ctk.CTkButton(self.navbar, text="Annotate", width=100, font=("Arial", 14), command=self.run_module_action)
        self.annotate_btn.pack(side="left", padx=5)

        # button for save annotation list as dictionary
        self.save_annotation_btn = ctk.CTkButton(self.navbar, text="Save Data", width=100, font=("Arial", 14), command=self.annotate_save_btn)
        self.save_annotation_btn.pack(side="left", padx=5)

        # status label for messages
        self.status_label = ctk.CTkLabel(self, text="", font=("Arial", 12))
        self.status_label.pack(side="bottom", pady=5)

        # video display (use Canvas for mouse events)
        self.video_canvas = ctk.CTkCanvas(self, highlightthickness=0, bg="black")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.annotator = Annotator()

        # GStreamer pipeline for USB webcam
        self.DEVICE_PATH = '/dev/video0'

        # Pipeline for MJPEG output
        self.pipeline = (
            f"v4l2src device={self.DEVICE_PATH} ! "
            "image/jpeg,width=1280,height=720,framerate=30/1 ! "
            "jpegdec ! video/x-raw ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink"
        )

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

        print("Press q to QUIT.")
        self.running = True
        self.screen_width = screen_width
        self.screen_height = screen_height - 80
        self.current_frame = None
        self.show_annotated = False
        self.is_annotating = False  # Flag to pause live feed during annotation
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.annotations = []
        self.update_video()

    def start_drawing(self, event):
        if not self.is_annotating:
            return  # Only allow drawing during annotation mode
        self.drawing = True

        # Frame dimensions
        frame_height, frame_width = self.current_frame.shape[:2]
        canvas_width = max(self.video_canvas.winfo_width(), 1)
        canvas_height = max(self.video_canvas.winfo_height(), 1)

        # Resize scale used in display_frame
        scale = min(self.screen_width / frame_width, self.screen_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        # Centering offsets
        offset_x = (canvas_width - new_width) // 2
        offset_y = (canvas_height - new_height) // 2

        # Adjust mouse coordinates relative to displayed frame
        adj_x = event.x - offset_x
        adj_y = event.y - offset_y

        # Clamp inside displayed frame
        adj_x = max(0, min(adj_x, new_width))
        adj_y = max(0, min(adj_y, new_height))

        # Scale back to original frame coordinates
        scale_x = frame_width / new_width
        scale_y = frame_height / new_height
        self.start_point = (int(adj_x * scale_x), int(adj_y * scale_y))

        print(f"Mouse clicked at: ({event.x}, {event.y}) → start_point: {self.start_point}")


    def update_drawing(self, event):
        if not self.is_annotating or not self.drawing:
            return

        frame_height, frame_width = self.current_frame.shape[:2]
        canvas_width = max(self.video_canvas.winfo_width(), 1)
        canvas_height = max(self.video_canvas.winfo_height(), 1)

        # Resize scale used in display_frame
        scale = min(self.screen_width / frame_width, self.screen_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        # Centering offsets
        offset_x = (canvas_width - new_width) // 2
        offset_y = (canvas_height - new_height) // 2

        # Adjust mouse coordinates relative to displayed frame
        adj_x = event.x - offset_x
        adj_y = event.y - offset_y

        # Clamp inside displayed frame
        adj_x = max(0, min(adj_x, new_width))
        adj_y = max(0, min(adj_y, new_height))

        # Scale back to original frame coordinates
        scale_x = frame_width / new_width
        scale_y = frame_height / new_height
        self.end_point = (int(adj_x * scale_x), int(adj_y * scale_y))

        print(f"Mouse dragging at: ({event.x}, {event.y}) → end_point: {self.end_point}")

        # Draw temporary rectangle for preview
        if self.current_frame is not None and self.start_point:
            temp_frame = self.current_frame.copy()
            cv2.rectangle(temp_frame, self.start_point, self.end_point, (0, 0, 255), 1)

            # Draw previously saved rectangles as well
            for rect in self.annotations:
                cv2.rectangle(temp_frame, (rect["x"], rect["y"]),
                            (rect["x"] + rect["width"], rect["y"] + rect["height"]),
                            (0, 0, 255), 1)

            self.display_frame(temp_frame)


    def stop_drawing(self, event):
        if not self.is_annotating:
            return
        self.drawing = False

        frame_height, frame_width = self.current_frame.shape[:2]
        canvas_width = max(self.video_canvas.winfo_width(), 1)
        canvas_height = max(self.video_canvas.winfo_height(), 1)

        # Resize scale used in display_frame
        scale = min(self.screen_width / frame_width, self.screen_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        # Centering offsets
        offset_x = (canvas_width - new_width) // 2
        offset_y = (canvas_height - new_height) // 2

        # Adjust mouse coordinates relative to displayed frame
        adj_x = event.x - offset_x
        adj_y = event.y - offset_y

        # Clamp inside displayed frame
        adj_x = max(0, min(adj_x, new_width))
        adj_y = max(0, min(adj_y, new_height))

        # Scale back to original frame coordinates
        scale_x = frame_width / new_width
        scale_y = frame_height / new_height
        self.end_point = (int(adj_x * scale_x), int(adj_y * scale_y))

        print(f"Mouse released at: ({event.x}, {event.y}) → end_point: {self.end_point}")

        if self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            rect = {
                "start": self.start_point,
                "end": self.end_point,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.annotations.append(rect)
            print(f"Rectangle saved: {rect}")

            # Refresh display with all accumulated rectangles
            temp_frame = self.current_frame.copy()
            for r in self.annotations:
                cv2.rectangle(temp_frame,
                            (r["x"], r["y"]),
                            (r["x"] + r["width"], r["y"] + r["height"]),
                            (0, 0, 255), 1)
            self.display_frame(temp_frame)


    def annotate_save_btn(self):
        """Save all accumulated annotations to a single JSON file."""
        if not self.annotations:
            print("No annotations to save.")
            return

        save_data = {
            "annotations": self.annotations,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # add the folder if exists
        save_dir = "annotation_logs"
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, "w") as f:
            json.dump(save_data, f, indent=4)

        print(f"Annotations saved to {filename}")
        time.sleep(1)
        self.resume_video()

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

    def display_frame(self, frame):
        # Resize frame based on screen size
        frame_height, frame_width = frame.shape[:2]
        scale = min(self.screen_width / frame_width, self.screen_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        frame = cv2.resize(frame, (new_width, new_height))

        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)

        # Update canvas
        self.video_canvas.delete("all")
        canvas_width = max(self.video_canvas.winfo_width(), 1)
        canvas_height = max(self.video_canvas.winfo_height(), 1)
        # print(f"Rendering on canvas: ({canvas_width}, {canvas_height})")
        self.video_canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
        self.video_canvas.image = photo

    def update_video(self):
        if not self.running or self.is_annotating:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            self.running = False
            self.cap.release()
            return

        # Upload frame to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Resize on GPU
        scale = min(self.screen_width / frame.shape[1], self.screen_height / frame.shape[0])
        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        gpu_resized = cv2.cuda.resize(gpu_frame, (new_width, new_height))

        # Download back to CPU (needed for Tkinter display)
        frame = gpu_resized.download()

        # No annotations drawn here

        self.current_frame = frame
        self.display_frame(frame)

        self.after(33, self.update_video)


    def destroy(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy()

if __name__ == "__main__":
    app = SharpeyeApp()
    app.protocol("WM_DELETE_WINDOW", app.destroy)
    app.bind('<q>', lambda event: app.destroy())
    app.mainloop()